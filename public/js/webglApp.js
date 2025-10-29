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
    // Generalized detection buffers (max 16)
    this.maxDetections = 16;
    this.nDetections = 0;
    this.detectionBoxes = new Float32Array(4 * this.maxDetections); // x,y,w,h in pixels
    this.detectionClassIds = new Float32Array(this.maxDetections);   // mapped float class ids
    this.detectionScores = new Float32Array(this.maxDetections);     // confidence [0,1]
    this.latestDetections = [];
    this.detectionInterval = null;
    this.detectionAdapter = null;
    this.log = null;
    this._fpsFrames = 0;
    this._fpsLastTs = performance.now();
    this._detectAvgMs = 0;
    this._detectCount = 0;
    this._logRender = false;
    this._logDetect = false;
  }

  async init() {
    this.initWebGL();
    await this.startWebcam();
    this.animate();
  }

  initWebGL() {
    this.gl = this.canvas.getContext('webgl');
    if (!this.gl) throw new Error('WebGL not supported');
    this.log?.('WebGL initialized');

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
      // Universal detection uniforms
      uniform float u_numDetections;              // number of active detections (0..16)
      uniform vec4 u_detectionBoxes[16];          // (x,y,w,h) in pixels
      uniform float u_detectionClassIds[16];      // class id per detection (float)
      uniform float u_detectionScores[16];        // confidence per detection
      uniform vec2 u_resolution;
      
      
      void main() {
        vec4 videoColor = texture2D(uVideo, vUv);
        vec3 finalColor = vec3(0.0);
        
        if (videoColor.a > 0.0) {
          // Base: use original video
          finalColor = videoColor.rgb;

          // Example default effect: highlight any detection by boosting saturation inside boxes
          if (u_numDetections > 0.0) {
            vec2 px = vUv * u_resolution;
            bool inAny = false;
            for (int i = 0; i < 16; i++) {
              if (float(i) >= u_numDetections) break;
              vec4 b = u_detectionBoxes[i];
              vec2 minB = b.xy;
              vec2 maxB = b.xy + b.zw;
              if (px.x >= minB.x && px.x <= maxB.x && px.y >= minB.y && px.y <= maxB.y) {
                inAny = true;
                break;
              }
            }

            if (inAny) {
              vec3 c = finalColor;
              float l = dot(c, vec3(0.299, 0.587, 0.114));
              finalColor = mix(vec3(l), c, 1.4);
              finalColor = clamp(finalColor, 0.0, 1.0);
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
    const t0 = performance.now();
    const newFrag = this.createShader(this.gl.FRAGMENT_SHADER, source);
    if (!newFrag) return false;
    if (this.program) {
      // Detach old, attach new, relink
      this.gl.detachShader(this.program, this.fragmentShader);
      this.gl.attachShader(this.program, newFrag);
      this.gl.linkProgram(this.program);
      this.gl.useProgram(this.program);
      this.fragmentShader = newFrag;
      const dt = Math.round(performance.now() - t0);
      this.log?.(`Shader compiled and linked in ${dt} ms`);
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
      this.log?.('Shader compile error (see console)');
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
    this.log?.('Webcam started');
  }

  enableML5() {
    if (this.ml5Enabled || !window.ml5) return;
    
    this.ml5Enabled = true;
    
    this.detector = ml5.objectDetector('cocossd', () => {
      this.log?.('ML5 COCO-SSD loaded');
      this.detectLoop();
    });
  }

  disableML5() {
    if (!this.ml5Enabled) return;
    
    this.ml5Enabled = false;
    // Clear detections when disabled
    this.nDetections = 0;
    
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
    
    const t0 = performance.now();
    this.detector.detect(this.video, (err, results) => {
      if (!err && results) {
        this.latestDetections = results;
        // Keep top N by confidence
        const sorted = [...results].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
        this.nDetections = Math.min(sorted.length, this.maxDetections);
        for (let i = 0; i < this.nDetections; i++) {
          const r = sorted[i]; // {label, confidence, x, y, width, height}
          const p = i * 4;
          this.detectionBoxes[p + 0] = r.x;
          this.detectionBoxes[p + 1] = r.y;
          this.detectionBoxes[p + 2] = r.width;
          this.detectionBoxes[p + 3] = r.height;
          // Default class id is 0; adapter can remap labels later
          this.detectionClassIds[i] = 0.0;
          this.detectionScores[i] = r.confidence || 0.0;
        }
        const dt = performance.now() - t0;
        // Exponential moving average for detection latency
        this._detectAvgMs = this._detectAvgMs ? (0.9 * this._detectAvgMs + 0.1 * dt) : dt;
        this._detectCount++;
        if (this._logDetect && this._detectCount % 20 === 0) {
          this.log?.(`Detection avg ${this._detectAvgMs.toFixed(1)} ms, n=${this.nDetections}`);
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
      
      // Upload detection uniforms through adapter if provided; else default
      if (typeof this.detectionAdapter === 'function') {
        this.detectionAdapter({
          gl: this.gl,
          program: this.program,
          // raw buffers
          nDetections: this.nDetections,
          detectionBoxes: this.detectionBoxes,
          detectionClassIds: this.detectionClassIds,
          detectionScores: this.detectionScores,
          latestDetections: this.latestDetections,
          resolution: [this.canvas.width, this.canvas.height],
        });
      } else {
        // Default mapping to universal uniforms
        const nLoc = this.gl.getUniformLocation(this.program, 'u_numDetections');
        this.gl.uniform1f(nLoc, this.nDetections);
        if (this.nDetections > 0) {
          const boxesLoc = this.gl.getUniformLocation(this.program, 'u_detectionBoxes');
          this.gl.uniform4fv(boxesLoc, this.detectionBoxes);
          const clsLoc = this.gl.getUniformLocation(this.program, 'u_detectionClassIds');
          this.gl.uniform1fv(clsLoc, this.detectionClassIds);
          const scrLoc = this.gl.getUniformLocation(this.program, 'u_detectionScores');
          this.gl.uniform1fv(scrLoc, this.detectionScores);
        }
      }

      
      this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    }
    requestAnimationFrame(() => this.animate());

    // FPS reporting ~1 Hz
    this._fpsFrames++;
    const now = performance.now();
    if (now - this._fpsLastTs >= 1000) {
      const fps = Math.round((this._fpsFrames * 1000) / (now - this._fpsLastTs));
      if (this._logRender) this.log?.(`Render FPS ${fps}, detections ${this.nDetections}`);
      this._fpsFrames = 0;
      this._fpsLastTs = now;
    }
  }
}

// Allow callers (or generated code) to override how detections map to shader uniforms
VideoEffectsApp.prototype.setDetectionAdapter = function(adapterFn) {
  this.detectionAdapter = adapterFn;
};

// Optional logger hookup
VideoEffectsApp.prototype.setLogger = function(logFn) {
  this.log = typeof logFn === 'function' ? logFn : null;
};

// Enable/disable verbose logging categories
VideoEffectsApp.prototype.setLoggingOptions = function(opts) {
  if (!opts || typeof opts !== 'object') return;
  if (typeof opts.render === 'boolean') this._logRender = opts.render;
  if (typeof opts.detection === 'boolean') this._logDetect = opts.detection;
};



