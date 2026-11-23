// ============================================================================
// SPOROCARP — 2048x2048 Directional PCF Shadow Map Pass (WebGPU)
// Complements FastSdfGi's soft penumbra & 2-bounce GI with crisp gill,
// torn-veil annulus, stipe, and cap-margin contact shadows.
// ============================================================================
import { mat4LookAt, mat4Multiply, type Mat4, type Vec3 } from './math.ts';

export const SHADOW_MAP_SIZE = 2048;

const SHADOW_WGSL = /* wgsl */ `
struct SceneUniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
  lightDir: vec4<f32>,
  soilBoundsMin: vec4<f32>,
  soilBoundsMax: vec4<f32>,
  sliceParams: vec4<f32>,
  sssColor: vec4<f32>,
  sporePrintColor: vec4<f32>,
  soilDims: vec4<f32>,
  sdfVolMin: vec4<f32>,
  sdfVolMax: vec4<f32>,
  camRight: vec4<f32>,
  camUp: vec4<f32>,
  lightViewProj: mat4x4<f32>,
  speciesExtra: vec4<f32>,
  postParams: vec4<f32>,
  padExtra: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;

struct ShadowOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
};

@vertex
fn vs_shadow(@location(0) pos: vec3<f32>) -> ShadowOut {
  var out: ShadowOut;
  out.clipPos = u.lightViewProj * vec4<f32>(pos, 1.0);
  out.worldPos = pos;
  return out;
}

@fragment
fn fs_shadow(@location(0) worldPos: vec3<f32>) {
  if (u.sliceParams.x > 0.5) {
    if (u.sliceParams.y < 0.5) {
      if (worldPos.x > u.sliceParams.z + 0.0005) { discard; }
    } else {
      if (worldPos.z > u.sliceParams.z + 0.0005) { discard; }
    }
  }
}
`;

export function mat4OrthoWebGPU(
  left: number,
  right: number,
  bottom: number,
  top: number,
  near: number,
  far: number,
  out: Mat4 = new Float32Array(16),
): Mat4 {
  out.fill(0);
  out[0] = 2 / (right - left);
  out[5] = 2 / (top - bottom);
  out[10] = 1 / (near - far);
  out[12] = -(right + left) / (right - left);
  out[13] = -(top + bottom) / (top - bottom);
  out[14] = near / (near - far);
  out[15] = 1;
  return out;
}

export class ShadowMapPass {
  readonly shadowTex: GPUTexture;
  readonly shadowView: GPUTextureView;
  readonly shadowSampler: GPUSampler;

  private mushPipe: GPURenderPipeline;
  private obsPipe: GPURenderPipeline;
  private soilPipe: GPURenderPipeline;

  private orthoMat = new Float32Array(16);
  private lightViewMat = new Float32Array(16);

  constructor(device: GPUDevice, simpleBindGroupLayout: GPUBindGroupLayout) {
    this.shadowTex = device.createTexture({
      size: [SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowView = this.shadowTex.createView();
    this.shadowSampler = device.createSampler({
      compare: 'less',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    const mod = device.createShaderModule({ code: SHADOW_WGSL });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [simpleBindGroupLayout] });

    const makePipe = (arrayStride: number, depthBias: number): GPURenderPipeline =>
      device.createRenderPipeline({
        layout,
        vertex: {
          module: mod,
          entryPoint: 'vs_shadow',
          buffers: [{
            arrayStride,
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
          }],
        },
        fragment: {
          module: mod,
          entryPoint: 'fs_shadow',
          targets: [],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: {
          format: 'depth32float',
          depthWriteEnabled: true,
          depthCompare: 'less',
          depthBias,
          depthBiasSlopeScale: 2.0,
        },
      });

    this.mushPipe = makePipe(64, 2);
    this.obsPipe = makePipe(40, 2);
    this.soilPipe = makePipe(28, 3);
  }

  computeLightViewProj(lightDir: Readonly<Vec3>, outMat: Float32Array): void {
    const l = lightDir;
    const len = Math.hypot(l[0], l[1], l[2]) || 1;
    const nx = l[0] / len;
    const ny = Math.max(0.18, l[1] / len);
    const nz = l[2] / len;
    const lLen = Math.hypot(nx, ny, nz);
    const eye: Vec3 = [(nx / lLen) * 0.95, (ny / lLen) * 0.95 + 0.02, (nz / lLen) * 0.95];
    const target: Vec3 = [0, 0.02, 0];
    const up: Vec3 = Math.abs(ny / lLen) > 0.94 ? [0, 0, 1] : [0, 1, 0];

    mat4LookAt(eye, target, up, this.lightViewMat);
    mat4OrthoWebGPU(-0.34, 0.34, -0.34, 0.34, 0.05, 2.0, this.orthoMat);
    mat4Multiply(this.orthoMat, this.lightViewMat, outMat);
  }

  render(
    encoder: GPUCommandEncoder,
    simpleBindGroup: GPUBindGroup,
    mushVertBuf: GPUBuffer,
    mushIdxBuf: GPUBuffer,
    mushIndexCount: number,
    obsVertBuf: GPUBuffer,
    obsVertCount: number,
    soilVertBuf: GPUBuffer,
    soilVertCount: number,
  ): void {
    const pass = encoder.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: this.shadowView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    if (soilVertCount > 0) {
      pass.setPipeline(this.soilPipe);
      pass.setBindGroup(0, simpleBindGroup);
      pass.setVertexBuffer(0, soilVertBuf);
      pass.draw(soilVertCount);
    }

    if (obsVertCount > 0) {
      pass.setPipeline(this.obsPipe);
      pass.setBindGroup(0, simpleBindGroup);
      pass.setVertexBuffer(0, obsVertBuf);
      pass.draw(obsVertCount);
    }

    if (mushIndexCount > 0) {
      pass.setPipeline(this.mushPipe);
      pass.setBindGroup(0, simpleBindGroup);
      pass.setVertexBuffer(0, mushVertBuf);
      pass.setIndexBuffer(mushIdxBuf, 'uint32');
      pass.drawIndexed(mushIndexCount);
    }

    pass.end();
  }

  dispose(): void {
    this.shadowTex.destroy();
  }
}
