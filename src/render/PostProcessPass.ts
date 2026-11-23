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
  params2: vec4<f32>, // x = near, y = far, z = blurDirX (or lightingMood on extract), w = blurDirY
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
  let thr = select(u.params0.z, min(u.params0.z, 0.68), u.params2.z > 2.5);
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
  params2: vec4<f32>, // x = near, y = far, z = lightingMood (0/1/2/3), w = realTimeSec
};

@group(0) @binding(0) var<uniform> u: PostUniforms;
@group(0) @binding(1) var hdrTex: texture_2d<f32>;
@group(0) @binding(2) var bloomTex: texture_2d<f32>;
@group(0) @binding(3) var depthMsaa: texture_depth_multisampled_2d;
@group(0) @binding(4) var linearSamp: sampler;

fn linearizeDepth(d: f32, nearZ: f32, farZ: f32) -> f32 {
  return (nearZ * farZ) / max(farZ - d * (farZ - nearZ), 1e-4);
}

fn evalCoc(distM: f32, focusM: f32, dofStrength: f32) -> f32 {
  return clamp((abs(distM - focusM) - 0.14) / 0.34, 0.0, 1.0) * dofStrength;
}

fn interleavedGradientNoise(pixelPos: vec2<f32>) -> f32 {
  let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
  return fract(magic.z * fract(dot(pixelPos, magic.xy)));
}

fn hashU32(xIn: u32) -> u32 {
  var x = xIn;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x;
}

fn grainNoise(pix: vec2<u32>, frameSeed: u32) -> f32 {
  let hA = hashU32(pix.x * 1973u + pix.y * 9277u + frameSeed * 26699u);
  let hB = hashU32(pix.x * 4999u + pix.y * 3571u + frameSeed * 14411u + 0x9e3779b9u);
  let uA = f32(hA & 0xffffu) * (1.0 / 65535.0);
  let uB = f32(hB & 0xffffu) * (1.0 / 65535.0);
  return (uA - uB) * 0.004;
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

  // Calibrated scientific X-Ray bypass: razor-sharp, no bloom, no DoF, no ACES/grain/tint
  if (xrayMode > 0.5) {
    return vec4<f32>(pow(clamp(baseHdr, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2)), 1.0);
  }

  var color = baseHdr;
  let mood = u.params2.z;
  let isNocturnal = mood > 2.5;

  // Depth-weighted 9-tap Vogel disk bokeh DoF in Natural + un-sliced view
  let dofStrength = u.params1.y;
  if (u.params1.w < 0.5 && dofStrength > 0.01) {
    let dDims = vec2<i32>(textureDimensions(depthMsaa));
    let coord = clamp(vec2<i32>(in.clipPos.xy), vec2<i32>(0), dDims - vec2<i32>(1));
    let rawDepth = textureLoad(depthMsaa, coord, 0);
    if (rawDepth < 0.9999) {
      let nearZ = u.params2.x;
      let farZ = u.params2.y;
      let focusM = u.params1.x;
      let centerDist = linearizeDepth(rawDepth, nearZ, farZ);
      let centerCoc = evalCoc(centerDist, focusM, dofStrength);

      if (centerCoc > 0.015) {
        let texel = vec2<f32>(u.params0.x, u.params0.y);
        let radiusPx = centerCoc * select(5.2, 6.8, isNocturnal);
        let rotAngle = interleavedGradientNoise(in.clipPos.xy) * 6.2831853;
        let goldenAngle = 2.39996323;

        var acc = baseHdr;
        var wSum = 1.0;

        for (var i = 0; i < 9; i = i + 1) {
          let rNorm = sqrt((f32(i) + 0.5) / 9.0);
          let theta = f32(i) * goldenAngle + rotAngle;
          let dir = vec2<f32>(cos(theta), sin(theta));
          let tapUv = clamp(in.uv + dir * (rNorm * radiusPx) * texel, vec2<f32>(0.0), vec2<f32>(1.0));
          let tapCoord = clamp(vec2<i32>(tapUv * vec2<f32>(dDims)), vec2<i32>(0), dDims - vec2<i32>(1));
          let tapRawDepth = textureLoad(depthMsaa, tapCoord, 0);
          let isSky = tapRawDepth >= 0.9999;
          let tapDist = select(linearizeDepth(tapRawDepth, nearZ, farZ), farZ, isSky);
          let tapCoc = select(evalCoc(tapDist, focusM, dofStrength), dofStrength, isSky);

          let depthDiff = tapDist - centerDist;
          let sameLayer = smoothstep(0.020, 0.003, abs(depthDiff));
          // Reject sharp in-focus foreground caps from leaking into background blur
          let fgCompat = smoothstep(0.03, 0.24, tapCoc);
          // Reject background leakage onto crisp hero silhouettes
          let bgCompat = smoothstep(0.04, 0.26, centerCoc) * clamp(1.0 - max(0.0, tapCoc - centerCoc) * 1.4, 0.10, 1.0);
          let dirCompat = select(bgCompat, fgCompat, depthDiff < 0.0);
          let w = max(sameLayer, dirCompat);

          let tapCol = textureSampleLevel(hdrTex, linearSamp, tapUv, 0.0).rgb;
          acc += tapCol * w;
          wSum += w;
        }

        let blurred = acc / max(wSum, 1e-4);
        color = mix(color, blurred, smoothstep(0.015, 0.22, centerCoc) * 0.85);
      }
    }
  }

  // Physical HDR bloom + mood grading + vignette + ACES filmic + sRGB + micro-grain
  let bloomBoost = select(select(1.0, 1.18, mood >= 1.5 && mood < 2.5), 1.65, isNocturnal);
  let bloom = textureSampleLevel(bloomTex, linearSamp, in.uv, 0.0).rgb;
  color += bloom * (u.params0.w * bloomBoost);

  let centered = in.uv - vec2<f32>(0.5);
  let r2 = dot(centered, centered);

  var mapped = vec3<f32>(0.0);
  if (mood >= 0.5 && mood < 1.5) {
    // Mood 1: Canopy warm golden filmic lift
    let vignette = 1.0 - r2 * 0.14;
    color *= vignette;
    color = color * vec3<f32>(1.04, 1.00, 0.91) * 1.03 + vec3<f32>(0.0024, 0.0017, 0.0008);
    mapped = acesFilmic(color);
  } else if (mood >= 1.5 && mood < 2.5) {
    // Mood 2: Dark-Field Backlit deep shadows & luminous rim contrast
    let vignette = 1.0 - r2 * 0.30;
    color *= vignette;
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let toe = smoothstep(0.0, 0.085, luma);
    let lumLift = 1.0 + smoothstep(0.22, 1.05, luma) * 0.24;
    let moodTint = mix(vec3<f32>(0.92, 0.96, 1.04), vec3<f32>(1.04, 1.00, 0.95), smoothstep(0.04, 0.45, luma));
    color = color * mix(0.66, 1.04, toe) * lumLift * moodTint;
    mapped = acesFilmic(color * 1.02);
  } else if (mood >= 2.5) {
    // Mood 3: Nocturnal dark-adapted photographic filmic lift + deep cool blue-black shadows
    let vignette = 1.0 - r2 * 0.28;
    color *= vignette;
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let shadowToe = smoothstep(0.0, 0.055, luma);
    let coolShadow = mix(vec3<f32>(0.76, 0.88, 1.14), vec3<f32>(1.0, 1.0, 1.0), smoothstep(0.015, 0.18, luma));
    color = color * mix(0.68, 1.0, shadowToe) * coolShadow + vec3<f32>(0.0005, 0.0012, 0.0030);
    mapped = acesFilmic(color * 1.18);
  } else {
    // Mood 0: Studio neutral contrast
    let vignette = 1.0 - r2 * 0.20;
    color *= vignette;
    mapped = acesFilmic(color * 0.96);
  }

  let srgb = pow(mapped, vec3<f32>(1.0 / 2.2));
  let pix = vec2<u32>(max(vec2<i32>(in.clipPos.xy), vec2<i32>(0)));
  let frameSeed = u32(max(u.params2.w, 0.0) * 60.0);
  let grain = grainNoise(pix, frameSeed);
  return vec4<f32>(clamp(srgb + vec3<f32>(grain), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
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
    lightingMood = 0,
    realTimeSec = 0,
  ): void {
    if (!this.bloomViewA || !this.bloomViewB || !this.extractBg || !this.blurHBg || !this.blurVBg || !this.compositeBg) {
      return;
    }

    const wrappedTime = ((realTimeSec % 600) + 600) % 600;
    const isNocturnal = lightingMood > 2.5;
    const u = this.uniformScratch;
    u[0] = 1 / Math.max(1, this.width);
    u[1] = 1 / Math.max(1, this.height);
    u[2] = isNocturnal ? 0.68 : 1.08; // bloom threshold
    u[3] = bloomStrength;
    u[4] = focusDistM;
    u[5] = dofStrength;
    u[6] = xrayIdx;
    u[7] = sliceEnabled ? 1.0 : 0.0;
    u[8] = 0.02;
    u[9] = 8.0;
    u[10] = lightingMood;
    u[11] = wrappedTime;
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
