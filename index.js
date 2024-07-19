const CANVAS_SIZE = 512;

export const xyOffset = 0;
export const squareStride = 4 * 2;

export const square = new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]);

const canvas = document.getElementById("canvas");
if (!navigator.gpu) {
  throw new Error("WebGPU not supported on this browser.");
}
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  throw new Error("No appropriate GPUAdapter found.");
}
const device = await adapter.requestDevice();
const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format: canvasFormat });

const vertexBuffer = device.createBuffer({
  label: "Cell vertices",
  size: square.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, square);
const vertexBufferLayout = {
  arrayStride: squareStride,
  attributes: [
    {
      format: "float32x2",
      offset: xyOffset,
      shaderLocation: 0,
    },
  ],
};

const textureArray = new Float32Array(CANVAS_SIZE * CANVAS_SIZE * 3);
const textureStorage = device.createBuffer({
  label: "Texture storage",
  size: textureArray.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
textureArray.fill(0.4);
device.queue.writeBuffer(textureStorage, 0, textureArray);

const frameValue = new Uint32Array([0]);
const frameUniform = device.createBuffer({
  label: "Frame uniform",
  size: 4,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(frameUniform, 0, frameValue);

const PLANE_FIXED_X = 1 << 24;
const PLANE_FIXED_Y = 2 << 24;
const PLANE_FIXED_Z = 3 << 24;
const SPHERE = 4 << 24;

const EMISSIVE = 1 << 24;
const REFLECTIVE = 2 << 24;
const DIFFUSE = 3 << 24;

function toU8(x) {
  return (x + 256) % 256;
}

function toU32(x, y, z) {
  return (toU8(x) << 16) + (toU8(y) << 8) + toU8(z);
}

const topPlane = [PLANE_FIXED_Y + toU32(0, 5, 0), EMISSIVE + toU32(255, 255, 255)];
const bottomPlane = [PLANE_FIXED_Y + toU32(0, -5, 0), EMISSIVE + toU32(200, 200, 200)];
const leftPlane = [PLANE_FIXED_X + toU32(5, 0, 0), EMISSIVE + toU32(255, 150, 255)];
const rightPlane = [PLANE_FIXED_X + toU32(-5, 0, 0), EMISSIVE + toU32(150, 255, 255)];
const backPlane = [PLANE_FIXED_Z + toU32(0, 0, 10), REFLECTIVE + toU32(170, 170, 170)];
const backbackPlane = [PLANE_FIXED_Z + toU32(0, 0, -2), EMISSIVE + toU32(255, 255, 150)];
const sphere = [SPHERE + toU32(3, 2, 6), REFLECTIVE + toU32(150, 150, 180)];
const sphere2 = [SPHERE + toU32(-3, 4, 8), EMISSIVE + toU32(255, 255, 0)];
const sphere3 = [SPHERE + toU32(0, 4, 8), EMISSIVE + toU32(255, 0, 255)];
const sphere4 = [SPHERE + toU32(3, 4, 8), EMISSIVE + toU32(0, 255, 255)];

const objectsArray = new Uint32Array([
  ...topPlane,
  ...bottomPlane,
  ...leftPlane,
  ...rightPlane,
  ...backPlane,
  ...backbackPlane,
  // ...sphere,
  ...sphere2,
  ...sphere3,
  ...sphere4,
]); // (object_type, x, y, z; surface_type, r, g, b)
const objectsBuffer = device.createBuffer({
  label: "Scene objects",
  size: objectsArray.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(objectsBuffer, 0, objectsArray);

const SPACE_BOUND = 1000000; // larger than the space objects are in

const displayShaderModule = device.createShaderModule({
  label: "Display shaders (vertex and fragment)",
  code: /* wgsl */ `
    struct VertexInput {
      @location(0) pos: vec2f,
    }

    struct VertexOutput {
      @builtin(position) pos: vec4f,
    }

    @group(0) @binding(0) var<storage, read_write> pixels: array<f32>;

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
      var output: VertexOutput;
      output.pos = vec4f(input.pos, 0, 1);
      return output;
    }

    @fragment
    fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let index = 3 * (u32(pos.x) + (${CANVAS_SIZE} - u32(pos.y)) * ${CANVAS_SIZE});
      // return vec4f(f32(index) / (512 * 512 * 3));
      // return vec4f((pos.x) / ${CANVAS_SIZE}, pos.y / ${CANVAS_SIZE}, 0, 1);
      return vec4f(pixels[index], pixels[index + 1], pixels[index + 2], 1);
    }
  `,
});

const WORKGROUP_SIZE = 8;
const TOLERANCE = 0.01;
const STEPS = 10000;
const rayMarchingShaderModule = device.createShaderModule({
  label: "Ray marching shader",
  code: /* wgsl */ `
    struct ObjectWithDistance {
      @location(0) index: u32,
      @location(1) distance: f32,
    }

    @group(0) @binding(0) var<storage, read_write> pixels: array<f32>;
    @group(0) @binding(1) var<storage> objects: array<u32>;
    @group(0) @binding(2) var<uniform> frame: u32;

    fn getI8AsF32(v: u32, offset: u32) -> f32 {
      return f32((((v & (u32(255) << offset)) >> offset) + 128) % 256) - 128;
    }

    fn getX(v: u32) -> f32 {
      return getI8AsF32(v, 16);
    }

    fn getY(v: u32) -> f32 {
      return getI8AsF32(v, 8);
    }

    fn getZ(v: u32) -> f32 {
      return getI8AsF32(v, 0);
    }

    fn getPos(v: u32) -> vec3f {
      return vec3f(getX(v), getY(v), getZ(v));
    }

    fn distanceToObject(pos: vec3f, i: u32) -> f32 {
      switch objects[i] & (255 << 24) {
        case ${PLANE_FIXED_X}: {
          return abs(getX(objects[i]) - pos.x);
        }
        case ${PLANE_FIXED_Y}: {
          return abs(getY(objects[i]) - pos.y);
        }
        case ${PLANE_FIXED_Z}: {
          return abs(getZ(objects[i]) - pos.z);
        }
        case ${SPHERE}: {
          return length(getPos(objects[i]) - pos) - 1;
        }
        default: {
          return ${SPACE_BOUND};
        }
      }
    }

    fn getNormal(direction: vec3f, position: vec3f, i: u32) -> vec3f {
      switch objects[i] & (255 << 24) {
        case ${PLANE_FIXED_X}: {
          return normalize(vec3f(position.x - getX(objects[i]), 0, 0));
        }
        case ${PLANE_FIXED_Y}: {
          return normalize(vec3f(0, position.y - getY(objects[i]), 0));
        }
        case ${PLANE_FIXED_Z}: {
          return normalize(vec3f(0, 0, position.z - getZ(objects[i])));
        }
        case ${SPHERE}: {
          return normalize(position - getPos(objects[i])); // Only works from outside the sphere
        }
        default: {
          return normalize(-direction);
        }
      }
    }

    fn reflect(direction: vec3f, position: vec3f, i: u32) -> vec3f {
      let normal = getNormal(direction, position, i);
      return direction - 2 * dot(direction, normal) * normal;
    }

    fn findClosestObject(pos: vec3f) -> ObjectWithDistance {
      var closestObject: ObjectWithDistance; 
      closestObject.distance = f32(${SPACE_BOUND});
      for (var i = u32(0); i < arrayLength(&objects); i += 2) {
        let di = distanceToObject(pos, i);
        if (di < closestObject.distance) {
          closestObject.distance = di;
          closestObject.index = i;
        }
      }
      return closestObject;
    }

    fn getColor(c: u32) -> vec4f {
      return vec4f(
        f32((c & (255 << 16)) >> 16) / 255,
        f32((c & (255 << 8)) >> 8) / 255,
        f32(c & 255) / 255,
        1
      );
    }

    fn randInner(seed: u32) -> u32 {
      return (seed * 1705829 + 16061) % 10909;
    }

    fn rand(seed: u32) -> f32 {
      return f32(randInner(randInner(seed))) / 10909;
    }

    fn getRandomDirection(position: vec3f, direction: vec3f, seed: u32, i: u32) -> vec3f {
      let theta = 2 * 3.14159 * rand(seed + (1 << 24));
      let phi = acos(2 * rand(seed + (2 << 24)) - 1);
      let randomVector = vec3f(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
      let normal = getNormal(direction, position, i);
      return sign(dot(randomVector, normal)) * randomVector;
    }

    fn castRay(pos: vec3f, dir: vec3f, seed: u32) -> vec4f {
      var result = vec4f(0, 0, 0, 0);
      var direction = dir;
      var position = pos;
      var color = vec4f(1, 1, 1, 1);
      var emitted = false;
      var closestObject: ObjectWithDistance;
      for (var i = u32(0); i < ${STEPS} && !emitted; i++) {
        closestObject = findClosestObject(position);
        if (closestObject.distance < ${TOLERANCE}) {
          switch objects[closestObject.index + 1] & (255 << 24) {
            case ${EMISSIVE}: {
              // return vec4f(0, 0, 0, 1);
              return color * getColor(objects[closestObject.index + 1]);
              emitted = true;
            }
            case ${REFLECTIVE}: {
              direction = reflect(direction, position, closestObject.index);
              color = color * getColor(objects[closestObject.index + 1]);
              position = position + direction * max(closestObject.distance, ${TOLERANCE});
            }
            case ${DIFFUSE}: {
              direction = getRandomDirection(position, direction, seed + bitcast<u32>(length(position * direction.yzx)), closestObject.index);
              color = color * getColor(objects[closestObject.index + 1]);
              position = position + direction * max(closestObject.distance, ${TOLERANCE});
            }
            default: {
              return vec4f(1, 0, 0, 1);
              emitted = true;
            }
          }
        // } else if (length(color) < 1.01) {
        //   return vec4f(0, 1, 0, 1);
        //   emitted = true;
        } else {
          position = position + direction * max(closestObject.distance, ${TOLERANCE});
        }
      }
      return vec4f(1, 0, 0, 1);
      // TODO: identify why we reach this situation so often
      // return vec4f(-position / 1, 1);

    }
    
    fn pxTo01(px: u32) -> f32 {
      return f32(2 * i32(px) - ${CANVAS_SIZE}) / ${CANVAS_SIZE};
    }

    @compute
    @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
    fn computeMain(@builtin(global_invocation_id) pixel: vec3u) {
      let direction = normalize(vec3f(pxTo01(pixel.x), pxTo01(pixel.y), 1));
      let position = vec3f(0, 0, 0);
      let index = 3 * (pixel.x + pixel.y * ${CANVAS_SIZE});
      let newColor = castRay(position, direction, index + frame);
      // let n = f32(frame);
      pixels[index] = newColor.x;
      pixels[index + 1] = newColor.y;
      pixels[index + 2] = newColor.z;
      // pixels[index] = pixels[index] * (n - 1) / n + newColor.x / n;
      // pixels[index + 1] = pixels[index + 1] * (n - 1) / n + newColor.y / n;
      // pixels[index + 2] = pixels[index + 2] * (n - 1) / n + newColor.z / n;
    }
  `,
});

const bindGroupLayout = device.createBindGroupLayout({
  label: "Bind group layout",
  entries: [
    {
      // Texture as floats
      binding: 0,
      visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    {
      // Scene objects
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" },
    },
    {
      // Frame
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {},
    },
  ],
});

const bindGroup = device.createBindGroup({
  label: "Bind group",
  layout: bindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: { buffer: textureStorage },
    },
    {
      binding: 1,
      resource: { buffer: objectsBuffer },
    },
    {
      binding: 2,
      resource: { buffer: frameUniform },
    },
  ],
});

const pipelineLayout = device.createPipelineLayout({
  label: "Display pipeline Layout",
  bindGroupLayouts: [bindGroupLayout],
});

const displayPipeline = device.createRenderPipeline({
  label: "Display pipeline",
  layout: pipelineLayout,
  vertex: {
    module: displayShaderModule,
    entryPoint: "vertexMain",
    buffers: [vertexBufferLayout],
  },
  fragment: {
    module: displayShaderModule,
    entryPoint: "fragmentMain",
    targets: [
      {
        format: canvasFormat,
      },
    ],
  },
});

const rayMarchingPipeline = device.createComputePipeline({
  label: "Ray marching pipeline",
  layout: pipelineLayout,
  compute: {
    module: rayMarchingShaderModule,
    entryPoint: "computeMain",
  },
});

function update() {
  frameValue[0] += 1;
  device.queue.writeBuffer(frameUniform, 0, frameValue);

  const encoder = device.createCommandEncoder();

  const computePass = encoder.beginComputePass();
  computePass.setPipeline(rayMarchingPipeline);
  computePass.setBindGroup(0, bindGroup);
  const workgroupCount = Math.ceil(CANVAS_SIZE / WORKGROUP_SIZE);
  computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
  computePass.end();

  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });

  pass.setPipeline(displayPipeline);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.setBindGroup(0, bindGroup);
  pass.draw(square.length / 2);
  pass.end();
  device.queue.submit([encoder.finish()]);

  // window.requestAnimationFrame(update);
}

update();
