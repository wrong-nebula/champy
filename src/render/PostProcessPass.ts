// ============================================================================
// SPOROCARP — HDR Post-Processing Pass (WebGPU)
// Pipeline:
//   - Main 4x MSAA rgba16float HDR buffer -> 1x rgba16float resolve target
//   - Half-res soft-knee HDR Bloom extract + 2-pass separable Gaussian blur
//   - Depth-aware Macro Bokeh DoF (read from 4x MSAA depth sample 0)
//   - Narkowicz ACES Filmic tone mapping + studio vignette + sRGB encode
//   (Automatically bypasses bloom/DoF/ACES in scientific X-Ray modes so
//    Viridis/Inferno colormaps remain razor-sharp and calibrated.)
// ============================================================================

const POST_VS_WGSL = /* wgsl */ `
struct FullVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) vid: u32) -> FullVertOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  let p = pos[vid];
  var out: FullVertOut;
  out.clipPos = vec4<f32>(p, 0.0, 1.0);
  // WebGPU texture coordinates have (0,0) top-left while clip +Y is up
  out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}
`;

const BLOOM_WGSL = /* wgsl */ `
${POST_VS_WGSL}

struct PostUniforms {
  params0: vec4<f32>, // x = texelW, y = texelH, z = bloomThreshold, w = bloomStrength
  params1: vec4<f32>, // x = focusDistM, y = dofStrength, z = xrayMode, w = sliceEnabled
  params2: vec4<f32>, // x = near, y = far, z = blurDirX, w = blurDirY
};

@group(0) @binding(0) var<uniform> u: PostUniforms;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var linearSamp: sampler;

@fragment
fn fs_bloom_extract(in: FullVertOut) -> @location(0) vec4<f32> {
  if (u.params1.z > 0.5) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let c = textureSampleLevel(srcTex, linearSamp, in.uv, 0.0).rgb;
  let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  let thr = u.params0.z;
  let knee = thr * 0.45;
  let soft = clamp(luma - thr + knee, 0.0, 2.0 * knee);
  let contrib = max(luma - thr, (soft * soft) / max(4.0 * knee, 1e-4));
  let scale = contrib / max(luma, 1e-4);
  return vec4<f32>(min(c * scale, vec3<f32>(4.0)), 1.0);
}

@fragment
fn fs_bloom_blur(in: FullVertOut) -> @location(0) vec4<f32> {
  let stepUv = vec2<f32>(u.params0.x * u.params2.z, u.params0.y * u.params2.w);
  let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
  var acc = textureSampleLevel(srcTex, linearSamp, in.uv, 0.0).rgb * weights[0];
  for (var i = 1; i < 5; i = i + 1) {
    let off = stepUv * (f32(i) * 1.45);
    acc += textureSampleLevel(srcTex, linearSamp, clamp(in.uv + off, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb * weights[i];
    acc += textureSampleLevel(srcTex, linearSamp, clamp(in.uv - off, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb * weights[i];
  }
  return vec4<f32>(acc, 1.0);
}
`;

const COMPOSITE_WGSL = /* wgsl */ `
${POST_VS_WGSL}

struct PostUniforms {
  params0: vec4<f32>, // x = texelW, y = texelH, z = bloomThreshold, w = bloomStrength
  params1: vec4<f32>, // x = focusDistM, y = dofStrength, z = xrayMode, w = sliceEnabled
  params2: vec4<f32>, // x = near, y = far, z = 0, w = 0
};

@group(0) @binding(0) var<uniform> u: PostUniforms;
@group(0) @binding(1) var hdrTex: texture_2d<f32>;
@group(0) @binding(2) var bloomTex: texture_2d<f32>;
@group(0) @binding(3) var depthMsaa: texture_depth_multisampled_2d;
@group(0) @binding(4) var linearSamp: sampler;

fn linearizeDepth(d: f32, nearZ: f32, farZ: f32) -> f32 {
  return (nearZ * farZ) / max(farZ - d * (farZ - nearZ), 1e-4);
}

fn acesFilmic(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp(
    (x * (a * x + vec3<f32>(b))) / (x * (c * x + vec3<f32>(d)) + vec3<f32>(e)),
    vec3<f32>(0.0),
    vec3<f32>(1.0),
  );
}

@fragment
fn fs_composite(in: FullVertOut) -> @location(0) vec4<f32> {
  let xrayMode = u.params1.z;
  let baseHdr = textureSampleLevel(hdrTex, linearSamp, in.uv, 0.0).rgb;

  // Calibrated scientific X-Ray bypass: razor-sharp, no bloom, no DoF, no ACES shift
  if (xrayMode > 0.5) {
    return vec4<f32>(pow(clamp(baseHdr, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2)), 1.0);
  }

  var color = baseHdr;

  // Gentle depth-aware macro bokeh DoF in Natural + un-sliced view
  let dofStrength = u.params1.y;
  if (u.params1.w < 0.5 && dofStrength > 0.01) {
    let dDims = vec2<i32>(textureDimensions(depthMsaa));
    let coord = clamp(vec2<i32>(in.clipPos.xy), vec2<i32>(0), dDims - vec2<i32>(1));
    let rawDepth = textureLoad(depthMsaa, coord, 0);
    if (rawDepth < 0.9999) {
      let distM = linearizeDepth(rawDepth, u.params2.x, u.params2.y);
      let focusM = u.params1.x;
      let coc = clamp((abs(distM - focusM) - 0.22) / 0.42, 0.0, 1.0) * dofStrength;
      if (coc > 0.02) {
        let px = vec2<f32>(u.params0.x, u.params0.y) * (coc * 2.2);
        let t0 = textureSampleLevel(hdrTex, linearSamp, clamp(in.uv + vec2<f32>(px.x, px.y), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
        let t1 = textureSampleLevel(hdrTex, linearSamp, clamp(in.uv + vec2<f32>(-px.x, px.y), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
        let t2 = textureSampleLevel(hdrTex, linearSamp, clamp(in.uv + vec2<f32>(px.x, -px.y), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
        let t3 = textureSampleLevel(hdrTex, linearSamp, clamp(in.uv + vec2<f32>(-px.x, -px.y), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
        let blurred = (color + t0 + t1 + t2 + t3) * 0.2;
        color = mix(color, blurred, coc * 0.60);
      }
    }
  }

  // Physical HDR bloom + subtle studio lens vignette + ACES filmic + sRGB
  let bloom = textureSampleLevel(bloomTex, linearSamp, in.uv, 0.0).rgb;
  color += bloom * u.params0.w;

  let centered = in.uv - vec2<f32>(0.5);
  let vignette = 1.0 - dot(centered, centered) * 0.20;
  color *= vignette;

  let mapped = acesFilmic(color * 0.96);
  let srgb = pow(mapped, vec3<f32>(1.0 / 2.2));
  return vec4<f32>(srgb, 1.0);
}
`;

export class PostProcessPass {
  readonly hdrFormat: GPUTextureFormat = 'rgba16float';
  private device: GPUDevice;
  private swapFormat: GPUTextureFormat;

  private width = 1;
  private height = 1;
  private halfW = 1;
  private halfH = 1;

  private hdrMsaaTex: GPUTexture | null = null;
  private hdrMsaaView: GPUTextureView | null = null;
  private hdrResolveTex: GPUTexture | null = null;
  private hdrResolveView: GPUTextureView | null = null;
  private bloomTexA: GPUTexture | null = null;
  private bloomViewA: GPUTextureView | null = null;
  private bloomTexB: GPUTexture | null = null;
  private bloomViewB: GPUTextureView | null = null;

  private uniformExtractBuf: GPUBuffer;
  private uniformBlurHBuf: GPUBuffer;
  private uniformBlurVBuf: GPUBuffer;
  private linearSampler: GPUSampler;
  private uniformScratch = new Float32Array(12);

  private extractPipe: GPURenderPipeline;
  private blurPipe: GPURenderPipeline;
  private compositePipe: GPURenderPipeline;

  private bloomLayout: GPUBindGroupLayout;
  private compositeLayout: GPUBindGroupLayout;

  private extractBg: GPUBindGroup | null = null;
  private blurHBg: GPUBindGroup | null = null;
  private blurVBg: GPUBindGroup | null = null;
  private compositeBg: GPUBindGroup | null = null;
  private cachedDepthView: GPUTextureView | null = null;

  constructor(device: GPUDevice, swapFormat: GPUTextureFormat) {
    this.device = device;
    this.swapFormat = swapFormat;

    this.uniformExtractBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uniformBlurHBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uniformBlurVBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.linearSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    const bloomMod = device.createShaderModule({ code: BLOOM_WGSL });
    this.bloomLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    const bloomPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bloomLayout] });
    this.extractPipe = device.createRenderPipeline({
      layout: bloomPipeLayout,
      vertex: { module: bloomMod, entryPoint: 'vs_full' },
      fragment: { module: bloomMod, entryPoint: 'fs_bloom_extract', targets: [{ format: this.hdrFormat }] },
      primitive: { topology: 'triangle-list' },
    });
    this.blurPipe = device.createRenderPipeline({
      layout: bloomPipeLayout,
      vertex: { module: bloomMod, entryPoint: 'vs_full' },
      fragment: { module: bloomMod, entryPoint: 'fs_bloom_blur', targets: [{ format: this.hdrFormat }] },
      primitive: { topology: 'triangle-list' },
    });

    const compMod = device.createShaderModule({ code: COMPOSITE_WGSL });
    this.compositeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', multisampled: true } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    this.compositePipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.compositeLayout] }),
      vertex: { module: compMod, entryPoint: 'vs_full' },
      fragment: { module: compMod, entryPoint: 'fs_composite', targets: [{ format: this.swapFormat }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  ensureTargets(width: number, height: number, depthView: GPUTextureView): void {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    const hw = Math.max(1, Math.floor(w / 2));
    const hh = Math.max(1, Math.floor(h / 2));

    if (
      w !== this.width ||
      h !== this.height ||
      !this.hdrMsaaTex ||
      !this.hdrResolveTex ||
      !this.bloomTexA ||
      !this.bloomTexB
    ) {
      this.width = w;
      this.height = h;
      this.halfW = hw;
      this.halfH = hh;

      this.hdrMsaaTex?.destroy();
      this.hdrResolveTex?.destroy();
      this.bloomTexA?.destroy();
      this.bloomTexB?.destroy();

      this.hdrMsaaTex = this.device.createTexture({
        size: [w, h, 1],
        sampleCount: 4,
        format: this.hdrFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.hdrMsaaView = this.hdrMsaaTex.createView();

      this.hdrResolveTex = this.device.createTexture({
        size: [w, h, 1],
        format: this.hdrFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.hdrResolveView = this.hdrResolveTex.createView();

      this.bloomTexA = this.device.createTexture({
        size: [hw, hh, 1],
        format: this.hdrFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.bloomViewA = this.bloomTexA.createView();

      this.bloomTexB = this.device.createTexture({
        size: [hw, hh, 1],
        format: this.hdrFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.bloomViewB = this.bloomTexB.createView();

      this.extractBg = this.device.createBindGroup({
        layout: this.bloomLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformExtractBuf } },
          { binding: 1, resource: this.hdrResolveView },
          { binding: 2, resource: this.linearSampler },
        ],
      });
      this.blurHBg = this.device.createBindGroup({
        layout: this.bloomLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBlurHBuf } },
          { binding: 1, resource: this.bloomViewA },
          { binding: 2, resource: this.linearSampler },
        ],
      });
      this.blurVBg = this.device.createBindGroup({
        layout: this.bloomLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBlurVBuf } },
          { binding: 1, resource: this.bloomViewB },
          { binding: 2, resource: this.linearSampler },
        ],
      });
      this.cachedDepthView = null;
    }

    if (this.cachedDepthView !== depthView && this.hdrResolveView && this.bloomViewA) {
      this.cachedDepthView = depthView;
      this.compositeBg = this.device.createBindGroup({
        layout: this.compositeLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformExtractBuf } },
          { binding: 1, resource: this.hdrResolveView },
          { binding: 2, resource: this.bloomViewA },
          { binding: 3, resource: depthView },
          { binding: 4, resource: this.linearSampler },
        ],
      });
    }
  }

  getHdrMsaaView(): GPUTextureView {
    return this.hdrMsaaView!;
  }

  getHdrResolveView(): GPUTextureView {
    return this.hdrResolveView!;
  }

  renderPost(
    encoder: GPUCommandEncoder,
    swapView: GPUTextureView,
    focusDistM: number,
    dofStrength: number,
    bloomStrength: number,
    xrayIdx: number,
    sliceEnabled: boolean,
  ): void {
    if (!this.bloomViewA || !this.bloomViewB || !this.extractBg || !this.blurHBg || !this.blurVBg || !this.compositeBg) {
      return;
    }

    const u = this.uniformScratch;
    u[0] = 1 / Math.max(1, this.width);
    u[1] = 1 / Math.max(1, this.height);
    u[2] = 1.08; // bloom threshold
    u[3] = bloomStrength;
    u[4] = focusDistM;
    u[5] = dofStrength;
    u[6] = xrayIdx;
    u[7] = sliceEnabled ? 1.0 : 0.0;
    u[8] = 0.02;
    u[9] = 8.0;
    u[10] = 0.0;
    u[11] = 0.0;
    this.device.queue.writeBuffer(this.uniformExtractBuf, 0, u.buffer, u.byteOffset, 48);

    u[0] = 1 / Math.max(1, this.halfW);
    u[1] = 1 / Math.max(1, this.halfH);
    u[10] = 1.0; u[11] = 0.0;
    this.device.queue.writeBuffer(this.uniformBlurHBuf, 0, u.buffer, u.byteOffset, 48);

    u[10] = 0.0; u[11] = 1.0;
    this.device.queue.writeBuffer(this.uniformBlurVBuf, 0, u.buffer, u.byteOffset, 48);

    // 1. Half-res bloom extract -> bloomTexA
    {
      const p = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.bloomViewA,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      p.setPipeline(this.extractPipe);
      p.setBindGroup(0, this.extractBg);
      p.draw(3);
      p.end();
    }

    // 2. Separable Gaussian horizontal blur -> bloomTexB
    {
      const p = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.bloomViewB,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      p.setPipeline(this.blurPipe);
      p.setBindGroup(0, this.blurHBg);
      p.draw(3);
      p.end();
    }

    // 3. Separable Gaussian vertical blur -> bloomTexA
    {
      const p = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.bloomViewA,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      p.setPipeline(this.blurPipe);
      p.setBindGroup(0, this.blurVBg);
      p.draw(3);
      p.end();
    }

    // 4. Composite DoF + Bloom + ACES Filmic + Vignette -> Swapchain
    {
      const p = encoder.beginRenderPass({
        colorAttachments: [{
          view: swapView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0.02, g: 0.03, b: 0.04, a: 1 },
        }],
      });
      p.setPipeline(this.compositePipe);
      p.setBindGroup(0, this.compositeBg);
      p.draw(3);
      p.end();
    }
  }

  dispose(): void {
    this.hdrMsaaTex?.destroy();
    this.hdrResolveTex?.destroy();
    this.bloomTexA?.destroy();
    this.bloomTexB?.destroy();
    this.uniformExtractBuf.destroy();
    this.uniformBlurHBuf.destroy();
    this.uniformBlurVBuf.destroy();
  }
}
