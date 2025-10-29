// OpenAI client: generates fragment shader from prompt and meta prompt
export async function generateShaderFromPrompt({ promptText, apiKeyInput, model, stateless, currentShader, onShader, onStatus }) {
  const apiKey = (apiKeyInput?.value || localStorage.getItem('openai_api_key') || '').trim();
  if (!apiKey) {
    alert('Please provide your OpenAI API key.');
    return;
  }

  const sys = await loadMetaPrompt();
  const user = `Make a fragment shader for this request. Keep it WebGL1-safe.\nRequest: ${promptText}`;

  const payload = {
    model: (model && typeof model === 'string' ? model : (localStorage.getItem('openai_model') || 'gpt-4o-mini')),
    messages: (stateless ? [
      { role: 'system', content: sys },
      { role: 'user', content: user },
    ] : [
      { role: 'system', content: sys },
      { role: 'user', content: [
          'Iterative mode: You will modify the provided fragment shader based on the user instruction. Maintain WebGL1 safety and the uniform/IO contract described in the system prompt. Return ONLY the full updated fragment shader source.',
          '\n\nCurrent shader:',
          '```glsl',
          (currentShader || ''),
          '```',
          '\n\nUser instruction:',
          promptText,
        ].join('\n') },
    ]),
    // temperature: 0.2,
    max_completion_tokens: 10000,
  };

  try {
    const t0 = performance.now();
    onStatus?.(`OpenAI: request started (model=${payload.model})`);
    const resp = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const msg = await resp.text();
      throw new Error(`OpenAI error: ${msg}`);
    }
    const t1 = performance.now();
    onStatus?.(`OpenAI: headers received in ${Math.round(t1 - t0)} ms`);
    const data = await resp.json();
    const t2 = performance.now();
    onStatus?.(`OpenAI: body parsed in ${Math.round(t2 - t1)} ms (total ${Math.round(t2 - t0)} ms)`);
    const content = data.choices?.[0]?.message?.content || '';
    const shader = extractCode(content);
    onStatus?.(`OpenAI: shader extracted (len=${shader.length})`);
    if (onShader) onShader(shader);
  } catch (e) {
    console.error(e);
    onStatus?.(`Error: ${String(e)}`);
    alert(String(e));
  }
}

export async function loadMetaPrompt() {
  try {
    const res = await fetch('./public/meta_prompt.txt');
    if (!res.ok) throw new Error('Failed to load meta_prompt.txt');
    const txt = await res.text();
    return txt.trim();
  } catch (e) {
    console.warn('Falling back to default meta prompt. Reason:', e);
    return [
      'You generate WebGL1-compatible fragment shaders only. Output ONLY the shader code.',
      'Requirements:',
      '- precision mediump float; must be included',
      '- Use varying vec2 vUv; and uniforms: sampler2D uVideo; float uTime; vec2 u_resolution;',
      '- Sample the video via texture2D(uVideo, vUv) and produce gl_FragColor',
      '- Do NOT include HTML/JS—just the fragment shader source',
    ].join('\n');
  }
}

function extractCode(text) {
  const codeBlock = text.match(/```[a-zA-Z]*\n([\s\S]*?)```/);
  return codeBlock ? codeBlock[1] : text;
}


