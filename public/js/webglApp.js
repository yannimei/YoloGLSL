// WebGL Video Effects App (ES module)
export class VideoEffectsApp {
  constructor({ videoSelector = '#video', canvasSelector = '#canvas' } = {}) {
    this.video = document.querySelector(videoSelector);
    this.canvas = document.querySelector(canvasSelector);
    this.gl = null;
    this.program = null;
    this.vertexShader = null;
    this.fragmentShader = null;
    this.videoTexture = null;
    this.startTime = Date.now();
    
    // ML5 properties
    this.ml5Enabled = false;
    this.detector = null;
    // CHANGED: maintain bottle detection boxes (x,y,w,h) up to 16
    this.bottleBoxes = new Float32Array(4 * 16);
    this.nBottles = 0;
    this.detectionInterval = null;
  }

  async init() {
    this.initWebGL();
    await this.startWebcam();
    this.animate();
  }

  initWebGL() {
    this.gl = this.canvas.getContext('webgl');
    if (!this.gl) throw new Error('WebGL not supported');

    // Default shaders
    const vertexSrc = `
      attribute vec2 position;
      varying vec2 vUv;
      void main() {
        vUv = position * 0.5 + 0.5;
        vUv.y = 1.0 - vUv.y; // Flip Y coordinate
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `;

    const fragmentSrc = `
      precision mediump float;
      varying vec2 vUv;
      uniform sampler2D uVideo;
      uniform float uTime;
      // CHANGED: only pass bottle detection boxes
      uniform float u_nbottles;
      uniform vec4 u_bottleBoxes[16]; // (x,y,w,h) in pixels
      uniform vec2 u_resolution;
      
      void main() {
        vec4 videoColor = texture2D(uVideo, vUv);
        vec3 finalColor = vec3(0.0);
        
        if (videoColor.a > 0.0) {
          // Base: use original video
          finalColor = videoColor.rgb;

          // CHANGED: Bottle detection - turn red colors to yellow
          if (u_nbottles > 0.0) {
            vec2 px = vUv * u_resolution;
            bool inBottle = false;
            for (int i = 0; i < 16; i++) {
              if (float(i) >= u_nbottles) break;
              vec4 b = u_bottleBoxes[i];
              vec2 minB = b.xy;
              vec2 maxB = b.xy + b.zw;
              if (px.x >= minB.x && px.x <= maxB.x && px.y >= minB.y && px.y <= maxB.y) {
                inBottle = true;
                break;
              }
            }

            if (inBottle) {
              // BOTTLE FILTER: Detect red colors and turn them yellow
              vec3 original = finalColor;
              
              // Check if pixel is red-ish (red channel dominant)
              float redness = original.r - max(original.g, original.b);
              if (redness > 0.1) { // Threshold for red detection
                // Convert red to yellow: keep red, boost green, reduce blue
                finalColor.r = original.r; // Keep red component
                finalColor.g = min(1.0, original.g + redness * 0.8); // Boost green
                finalColor.b = max(0.0, original.b - redness * 0.3); // Reduce blue slightly
              }
            }
          }
        } else {
          // Fallback pattern
          finalColor = vec3(vUv, 0.5 + 0.5 * sin(uTime));
        }
        
        gl_FragColor = vec4(finalColor, videoColor.a);
      }
    `;

    this.vertexShader = this.createShader(this.gl.VERTEX_SHADER, vertexSrc);
    this.fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, fragmentSrc);

    this.program = this.gl.createProgram();
    this.gl.attachShader(this.program, this.vertexShader);
    this.gl.attachShader(this.program, this.fragmentShader);
    this.gl.linkProgram(this.program);
    this.gl.useProgram(this.program);

    this.setupGeometry();
    this.setupVideoTexture();
  }

  // Allow dynamic replacement of fragment shader source
  setFragmentShader(source) {
    const newFrag = this.createShader(this.gl.FRAGMENT_SHADER, source);
    if (!newFrag) return false;
    if (this.program) {
      // Detach old, attach new, relink
      this.gl.detachShader(this.program, this.fragmentShader);
      this.gl.attachShader(this.program, newFrag);
      this.gl.linkProgram(this.program);
      this.gl.useProgram(this.program);
      this.fragmentShader = newFrag;
      return true;
    }
    return false;
  }

  createShader(type, source) {
    const shader = this.gl.createShader(type);
    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', this.gl.getShaderInfoLog(shader));
      this.gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  setupGeometry() {
    const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const buffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);
    const positionLocation = this.gl.getAttribLocation(this.program, 'position');
    this.gl.enableVertexAttribArray(positionLocation);
    this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 0, 0);
  }

  setupVideoTexture() {
    this.videoTexture = this.gl.createTexture();
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.videoTexture);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
  }

  async startWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    this.video.srcObject = stream;
    await new Promise(resolve => {
      this.video.onloadedmetadata = () => {
        this.video.play();
        resolve();
      };
    });
  }

  enableML5() {
    if (this.ml5Enabled || !window.ml5) return;
    
    console.log('Starting ML5 object detection...');
    this.ml5Enabled = true;
    
    this.detector = ml5.objectDetector('cocossd', () => {
      console.log('ML5 detector ready');
      this.detectLoop();
    });
  }

  disableML5() {
    if (!this.ml5Enabled) return;
    
    console.log('Disabling ML5 detection...');
    this.ml5Enabled = false;
    // CHANGED: clear bottle count when disabled
    this.nBottles = 0;
    
    if (this.detectionInterval) {
      clearTimeout(this.detectionInterval);
      this.detectionInterval = null;
    }
    
    if (this.detector) {
      this.detector = null;
    }
  }

  detectLoop = () => {
    if (!this.ml5Enabled || !this.detector || !this.video) return;
    
    this.detector.detect(this.video, (err, results) => {
      if (!err && results) {
        // CHANGED: detect bottles for red-to-yellow color effect
        const bottles = results.filter(r => r.label && r.label.toLowerCase() === 'bottle');
        this.nBottles = Math.min(bottles.length, 16);
        for (let i = 0; i < this.nBottles; i++) {
          const r = bottles[i]; // {label, confidence, x, y, width, height}
          const p = i * 4;
          this.bottleBoxes[p + 0] = r.x;
          this.bottleBoxes[p + 1] = r.y;
          this.bottleBoxes[p + 2] = r.width;
          this.bottleBoxes[p + 3] = r.height;
        }
      }
      
      // Throttle to ~10 FPS to save CPU
      this.detectionInterval = setTimeout(this.detectLoop, 100);
    });
  }

  animate() {
    if (this.gl && this.program) {
      this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      this.gl.clear(this.gl.COLOR_BUFFER_BIT);
      
      if (this.video.videoWidth > 0 && this.video.videoHeight > 0) {
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.videoTexture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, this.video);
      }
      
      // Set uniforms
      const timeLocation = this.gl.getUniformLocation(this.program, 'uTime');
      this.gl.uniform1f(timeLocation, (Date.now() - this.startTime) * 0.001);
      
      const videoLocation = this.gl.getUniformLocation(this.program, 'uVideo');
      this.gl.uniform1i(videoLocation, 0);
      
      const resolutionLocation = this.gl.getUniformLocation(this.program, 'u_resolution');
      this.gl.uniform2f(resolutionLocation, this.canvas.width, this.canvas.height);
      
      // CHANGED: upload bottle detection boxes
      const nBottlesLoc = this.gl.getUniformLocation(this.program, 'u_nbottles');
      this.gl.uniform1f(nBottlesLoc, this.nBottles);
      if (this.nBottles > 0) {
        const bottlesLoc = this.gl.getUniformLocation(this.program, 'u_bottleBoxes');
        this.gl.uniform4fv(bottlesLoc, this.bottleBoxes);
      }
      
      this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    }
    requestAnimationFrame(() => this.animate());
  }
}


