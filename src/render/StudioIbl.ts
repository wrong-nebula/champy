// ============================================================================
// SPOROCARP — Startup-Baked HDR Studio IBL Cubemap (WebGPU)
// Bakes a 128x128x6 rgba16float HDR cubemap (5 mip roughness levels 0..4)
// once on construction in ~1-2 ms using standard WebGPU core features only:
//   - Overhead warm-neutral photographic studio softbox (#fff7ea)
//   - Cool studio sky dome fill
//   - Subtle muted forest-green mid-horizon bounce
//   - Dark slate obsidian laboratory floor
//   - Subtle warm-amber studio kicker rim (+ soft cool opposing fill strip)
// ============================================================================

const CUBE_SIZE = 128;
const MIP_COUNT = 5;
const FACE_COUNT = 6;
const TOTAL_PASSES = MIP_COUNT * FACE_COUNT; // 30 passes

const STUDIO_IBL_WGSL = /* wgsl */ `
struct BakeUniforms {
  params: vec4<f32>, // x = face (0..5), y = roughness (0..1), z = mip (0..4), w = 0
};

@group(0) @binding(0) var<uniform> uBake: BakeUniforms;

struct FullVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> FullVertOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  let p = pos[vid];
  var out: FullVertOut;
  out.clipPos = vec4<f32>(p, 0.0, 1.0);
  // WebGPU framebuffer origin (0,0) is top-left while clip +Y is up
  out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

// Standard WebGPU / Vulkan cubemap face basis mapping from top-left UV in [0,1]^2
fn faceUvToDir(face: i32, uv: vec2<f32>) -> vec3<f32> {
  let sx = uv.x * 2.0 - 1.0;
  let sy = uv.y * 2.0 - 1.0;
  var d = vec3<f32>(0.0, 1.0, 0.0);
  if (face == 0) {
    d = vec3<f32>(1.0, -sy, -sx);      // +X
  } else if (face == 1) {
    d = vec3<f32>(-1.0, -sy, sx);      // -X
  } else if (face == 2) {
    d = vec3<f32>(sx, 1.0, sy);        // +Y
  } else if (face == 3) {
    d = vec3<f32>(sx, -1.0, -sy);      // -Y
  } else if (face == 4) {
    d = vec3<f32>(sx, -sy, 1.0);       // +Z
  } else {
    d = vec3<f32>(-sx, -sy, -1.0);     // -Z
  }
  return normalize(d);
}

// Botanical macro-photography studio environment (linear HDR)
fn evalStudioEnv(dir: vec3<f32>) -> vec3<f32> {
  let y = dir.y;

  // 1. Dark slate obsidian laboratory floor vs cool studio sky dome fill
  let nadirFloor = vec3<f32>(0.011, 0.015, 0.022);
  let horizonDark = vec3<f32>(0.030, 0.038, 0.048);
  let horizonSky = vec3<f32>(0.068, 0.092, 0.128);
  let zenithSky = vec3<f32>(0.138, 0.184, 0.252);

  let floorCol = mix(nadirFloor, horizonDark, smoothstep(-0.95, 0.0, y));
  let skyCol = mix(horizonSky, zenithSky, smoothstep(0.0, 0.92, y));
  var env = mix(floorCol, skyCol, smoothstep(-0.06, 0.10, y));

  // 2. Subtle muted forest-green mid-horizon botanical bounce
  let horizonBand = exp(-pow((y - 0.05) * 4.6, 2.0));
  let mossBounce = vec3<f32>(0.044, 0.078, 0.038);
  env = mix(env, env + mossBounce, horizonBand * 0.72);

  // 3. Overhead warm-neutral studio softbox (#fff7ea -> linear ~(1.0, 0.9315, 0.8276))
  let boxCenter = normalize(vec3<f32>(0.22, 0.92, 0.28));
  let boxRight = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), boxCenter));
  let boxFwd = cross(boxCenter, boxRight);
  let boxForwardDot = dot(dir, boxCenter);
  if (boxForwardDot > 0.0) {
    let px = dot(dir, boxRight) / max(boxForwardDot, 0.20);
    let pz = dot(dir, boxFwd) / max(boxForwardDot, 0.20);
    // Rounded-rectangle photographic softbox diffuser profile
    let qx = abs(px) / 0.62;
    let qz = abs(pz) / 0.46;
    let rectDist = pow(qx, 4.0) + pow(qz, 4.0);
    let softboxMask = 1.0 - smoothstep(0.65, 1.15, rectDist);
    let centerHot = 1.0 - 0.28 * clamp(px * px + pz * pz, 0.0, 1.0);
    let softboxCol = vec3<f32>(1.0, 0.9315, 0.8276) * (2.95 * centerHot);
    // Subtle dark light-skirt framing around the softbox edge
    let skirtMask = (1.0 - smoothstep(1.05, 1.65, rectDist)) * (1.0 - softboxMask);
    env = mix(env, env * 0.45, skirtMask * 0.65);
    env += softboxCol * softboxMask;
  }

  // 4. Subtle warm-amber studio kicker rim (low rear-left strip light)
  let kickerDir = normalize(vec3<f32>(-0.74, 0.28, -0.61));
  let kickerRight = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), kickerDir));
  let kickerUp = cross(kickerDir, kickerRight);
  let kickDot = dot(dir, kickerDir);
  if (kickDot > 0.0) {
    let kx = dot(dir, kickerRight) / max(kickDot, 0.25);
    let ky = dot(dir, kickerUp) / max(kickDot, 0.25);
    let kickDist = pow(abs(kx) / 0.26, 4.0) + pow(abs(ky) / 0.42, 4.0);
    let kickMask = 1.0 - smoothstep(0.55, 1.20, kickDist);
    let amberCol = vec3<f32>(1.0, 0.66, 0.28) * 1.45;
    env += amberCol * kickMask;
  }

  // 5. Gentle cool studio fill edge strip (opposite low side for crisp glass/cuticle separation)
  let fillDir = normalize(vec3<f32>(0.78, 0.22, -0.58));
  let fillDot = max(dot(dir, fillDir), 0.0);
  let fillMask = smoothstep(0.84, 0.96, fillDot);
  env += vec3<f32>(0.36, 0.54, 0.82) * (fillMask * 0.52);

  return env;
}

fn radicalInverseVdC(bitsIn: u32) -> f32 {
  var bits = (bitsIn << 16u) | (bitsIn >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
  return vec2<f32>(f32(i) / f32(n), radicalInverseVdC(i));
}

fn importanceSampleGGX(xi: vec2<f32>, N: vec3<f32>, roughness: f32) -> vec3<f32> {
  let a = roughness * roughness;
  let phi = 6.28318530718 * xi.x;
  let cosTheta = sqrt((1.0 - xi.y) / max(1.0 + (a * a - 1.0) * xi.y, 1e-5));
  let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

  let hLocal = vec3<f32>(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
  let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(N.z) < 0.999);
  let tangent = normalize(cross(up, N));
  let bitangent = cross(N, tangent);
  return normalize(tangent * hLocal.x + bitangent * hLocal.y + N * hLocal.z);
}

@fragment
fn fs_main(in: FullVertOut) -> @location(0) vec4<f32> {
  let face = i32(uBake.params.x + 0.5);
  let roughness = clamp(uBake.params.y, 0.0, 1.0);
  let N = faceUvToDir(face, in.uv);

  // Mip 0: sharp mirror reflection
  if (roughness < 0.02) {
    return vec4<f32>(evalStudioEnv(N), 1.0);
  }

  // Mips 1..4: GGX importance-sampled spherical prefilter convolution
  let numSamples = select(48u, 64u, roughness > 0.6);
  var prefiltered = vec3<f32>(0.0);
  var totalWeight = 0.0;
  let rClamped = max(roughness, 0.08);
  for (var i = 0u; i < numSamples; i = i + 1u) {
    let xi = hammersley(i, numSamples);
    let H = importanceSampleGGX(xi, N, rClamped);
    let L = normalize(2.0 * dot(N, H) * H - N);
    let ndotl = max(dot(N, L), 0.0);
    if (ndotl > 0.0) {
      prefiltered += evalStudioEnv(L) * ndotl;
      totalWeight += ndotl;
    }
  }
  let color = prefiltered / max(totalWeight, 1e-4);
  return vec4<f32>(color, 1.0);
}
`;

export class StudioIbl {
  readonly cubeView: GPUTextureView;
  readonly sampler: GPUSampler;

  private cubeTex: GPUTexture;
  private bakeUniformBuf: GPUBuffer;

  constructor(device: GPUDevice) {
    this.cubeTex = device.createTexture({
      size: [CUBE_SIZE, CUBE_SIZE, FACE_COUNT],
      mipLevelCount: MIP_COUNT,
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.cubeView = this.cubeTex.createView({
      dimension: 'cube',
      baseMipLevel: 0,
      mipLevelCount: MIP_COUNT,
      baseArrayLayer: 0,
      arrayLayerCount: FACE_COUNT,
    });

    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });

    const align = Math.max(256, device.limits.minUniformBufferOffsetAlignment || 256);
    const uniformData = new Float32Array((TOTAL_PASSES * align) / 4);
    this.bakeUniformBuf = device.createBuffer({
      size: TOTAL_PASSES * align,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let passIdx = 0;
    for (let mip = 0; mip < MIP_COUNT; mip++) {
      const roughness = mip / (MIP_COUNT - 1);
      for (let face = 0; face < FACE_COUNT; face++) {
        const fOff = (passIdx * align) / 4;
        uniformData[fOff] = face;
        uniformData[fOff + 1] = roughness;
        uniformData[fOff + 2] = mip;
        uniformData[fOff + 3] = 0;
        passIdx++;
      }
    }
    device.queue.writeBuffer(this.bakeUniformBuf, 0, uniformData);

    const mod = device.createShaderModule({ code: STUDIO_IBL_WGSL });
    const bgLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 16 },
      }],
    });
    const bakeBg = device.createBindGroup({
      layout: bgLayout,
      entries: [{
        binding: 0,
        resource: { buffer: this.bakeUniformBuf, offset: 0, size: 16 },
      }],
    });
    const pipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgLayout] }),
      vertex: { module: mod, entryPoint: 'vs_main' },
      fragment: { module: mod, entryPoint: 'fs_main', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });

    const encoder = device.createCommandEncoder();
    passIdx = 0;
    for (let mip = 0; mip < MIP_COUNT; mip++) {
      for (let face = 0; face < FACE_COUNT; face++) {
        const faceView = this.cubeTex.createView({
          dimension: '2d',
          baseMipLevel: mip,
          mipLevelCount: 1,
          baseArrayLayer: face,
          arrayLayerCount: 1,
        });
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: faceView,
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          }],
        });
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bakeBg, [passIdx * align]);
        pass.draw(3);
        pass.end();
        passIdx++;
      }
    }
    device.queue.submit([encoder.finish()]);
  }

  dispose(): void {
    this.cubeTex.destroy();
    this.bakeUniformBuf.destroy();
  }
}
