require('dotenv').config();
const { Bot, InputFile } = require('grammy');
const axios = require('axios');
const fs = require('fs/promises');
const fsSync = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const { execFile } = require('child_process');
const { promisify } = require('util');
const execFileAsync = promisify(execFile);

const bot = new Bot(process.env.TELEGRAM_TOKEN);

// Track which "mode" each chat is currently in
const userModes = {}; // chatId -> 'ai' | 'research' | 'web' | 'music' | 'videos' | 'images' | 'enhance_image' | 'enhance_video'

const mainKeyboard = {
  reply_markup: {
    keyboard: [
      ['🤖 AI Chat', '🧠 GPT (AIML)'],
      ['🔎 Research', '🔵 Web'],
      ['🎵 Music', '📺 Videos'],
      ['🖼️ Images', '✨ Enhance Image'],
      ['📽️ Enhance Video'],
    ],
    resize_keyboard: true,
  },
};

// ---------------------------------------------------------------------------
// Free video + music generation via Hugging Face's Serverless Inference API.
// No billing account required — just a free HF account and access token
// (huggingface.co/settings/tokens, read scope is enough).
//
// Trade-offs vs a paid provider (Kling/Runway/MiniMax):
//   - Rate-limited (a few hundred requests/hour, shared pool)
//   - Cold starts: first request after inactivity can 503 while the model
//     spins up — this code retries automatically when that happens
//   - Lower quality / shorter clips than paid models, and can fail outright
//     under load — that's expected on the free tier, not a bug
//
// Requires in .env:
//   HUGGINGFACE_API_TOKEN
// ---------------------------------------------------------------------------
const HF_VIDEO_MODEL = process.env.HF_VIDEO_MODEL || 'damo-vilab/text-to-video-ms-1.7b';
const HF_MUSIC_MODEL = process.env.HF_MUSIC_MODEL || 'facebook/musicgen-small';

async function callHfInference(model, prompt, { maxRetries = 4, maxWaitMs = 30000 } = {}) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const response = await axios.post(
      `https://router.huggingface.co/hf-inference/models/${model}`,
      { inputs: prompt },
      {
        headers: {
          Authorization: `Bearer ${process.env.HUGGINGFACE_API_TOKEN}`,
          'Content-Type': 'application/json',
        },
        responseType: 'arraybuffer',
        validateStatus: () => true, // handle non-200s ourselves
        timeout: 120000,
      }
    );

    if (response.status === 200) {
      return Buffer.from(response.data);
    }

    // Try to parse the JSON error body (HF sends JSON even though we asked
    // for arraybuffer, since the response isn't actually binary on errors).
    let errJson = null;
    try {
      errJson = JSON.parse(Buffer.from(response.data).toString('utf8'));
    } catch (_) {
      // not JSON, ignore
    }

    if (response.status === 503) {
      // Model is cold-starting. HF tells us roughly how long to wait.
      const waitMs = Math.min((errJson?.estimated_time || 15) * 1000, maxWaitMs);
      await new Promise((resolve) => setTimeout(resolve, waitMs));
      continue;
    }

    if (response.status === 429) {
      throw new Error('Rate-limited by Hugging Face\'s free tier — wait a bit and try again.');
    }

    throw new Error(errJson?.error || `Hugging Face API error (status ${response.status})`);
  }
  throw new Error('Model is still loading after multiple retries — try again in a minute.');
}

async function generateVideo(prompt) {
  return callHfInference(HF_VIDEO_MODEL, prompt, { maxRetries: 4, maxWaitMs: 30000 });
}

async function generateMusic(prompt) {
  return callHfInference(HF_MUSIC_MODEL, prompt, { maxRetries: 4, maxWaitMs: 30000 });
}

// ---------------------------------------------------------------------------
// Web / Research via SerpAPI (Google results).
//
// Requires in .env:
//   SERPAPI_API_KEY   (serpapi.com/manage-api-key)
// ---------------------------------------------------------------------------
async function serpApiSearch(query, num = 6) {
  const response = await axios.get('https://serpapi.com/search.json', {
    params: {
      q: query,
      num,
      api_key: process.env.SERPAPI_API_KEY,
    },
    timeout: 20000,
  });

  const data = response.data;
  if (data.error) throw new Error(data.error);

  const results = [];

  // Instant "answer box" content, when Google gives one directly.
  if (data.answer_box) {
    const ab = data.answer_box;
    const snippet = ab.answer || ab.snippet || ab.result;
    if (snippet) results.push({ title: ab.title || 'Featured answer', link: ab.link || '', snippet });
  }

  for (const r of data.organic_results || []) {
    results.push({ title: r.title, link: r.link, snippet: r.snippet || '' });
    if (results.length >= num) break;
  }

  return results;
}

function formatSearchResults(results) {
  if (!results.length) return 'No results found.';
  return results
    .map((r, i) => `${i + 1}. *${escapeMarkdown(r.title)}*\n${escapeMarkdown(r.snippet)}\n${r.link}`)
    .join('\n\n');
}

function escapeMarkdown(text = '') {
  // Minimal escaping for Telegram legacy Markdown parse mode.
  return text.replace(/([_*[\]`])/g, '\\$1');
}

async function askClaude(prompt, { maxTokens = 1024 } = {}) {
  const response = await axios.post(
    'https://api.anthropic.com/v1/messages',
    {
      model: 'claude-sonnet-5',
      max_tokens: maxTokens,
      messages: [{ role: 'user', content: prompt }],
    },
    {
      headers: {
        'x-api-key': process.env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
      },
    }
  );
  return response.data.content
    .filter((block) => block.type === 'text')
    .map((block) => block.text)
    .join('\n');
}

async function researchTopic(query) {
  const results = await serpApiSearch(query, 8);
  if (!results.length) return 'No results found for that — try rephrasing.';

  const sourcesBlock = results
    .map((r, i) => `[${i + 1}] ${r.title}\n${r.snippet}\nURL: ${r.link}`)
    .join('\n\n');

  const synthesis = await askClaude(
    `You are a research assistant. Using ONLY the sources below, write a concise, well-organized answer ` +
      `to the question: "${query}"\n\nCite sources inline using [1], [2], etc. matching the numbering below. ` +
      `If the sources don't fully answer the question, say so.\n\nSOURCES:\n${sourcesBlock}`,
    { maxTokens: 1024 }
  );

  const sourceList = results.map((r, i) => `[${i + 1}] ${r.link}`).join('\n');
  return `${synthesis}\n\n*Sources:*\n${sourceList}`;
}

// ---------------------------------------------------------------------------
// Image generation via Pollinations.ai — free, unlimited, no API key or
// signup required (Flux model). Just an HTTP GET.
// ---------------------------------------------------------------------------
async function generateImage(prompt) {
  const seed = Math.floor(Math.random() * 1e9); // avoid Cloudflare-cached repeats
  const url = `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}`;
  const response = await axios.get(url, {
    params: { width: 1024, height: 1024, nologo: true, seed, model: 'flux' },
    responseType: 'arraybuffer',
    timeout: 120000,
  });
  return Buffer.from(response.data);
}

// ---------------------------------------------------------------------------
// Image / video enhancement via a locally-run Real-ESRGAN binary — genuinely
// free (no per-call cost, no rate limit), at the cost of running on your own
// machine's CPU/GPU instead of a hosted API.
//
// ONE-TIME SETUP (not covered by npm install):
//   1. Download the portable "realesrgan-ncnn-vulkan" build for your OS from:
//        https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases
//   2. Unzip it into a `bin/` folder next to bot.js (or anywhere you like).
//   3. On Linux/macOS: chmod +x bin/realesrgan-ncnn-vulkan
//   4. Install ffmpeg (needed for video frame extraction/reassembly):
//        Ubuntu/Debian: sudo apt install ffmpeg
//        macOS:         brew install ffmpeg
//   5. (Optional) In .env, override the defaults below if your binary/model
//      live somewhere else:
//        REALESRGAN_BIN=./bin/realesrgan-ncnn-vulkan
//        REALESRGAN_MODEL=realesrgan-x4plus
//        FFMPEG_BIN=ffmpeg
//        FFPROBE_BIN=ffprobe
//
// No GPU? The ncnn-vulkan build still runs on CPU via software Vulkan, just
// slower — expect several seconds per image, and video enhancement can take
// minutes since every frame is upscaled individually.
// ---------------------------------------------------------------------------
const REALESRGAN_BIN = process.env.REALESRGAN_BIN || path.join(__dirname, 'bin', 'realesrgan-ncnn-vulkan');
const REALESRGAN_MODEL = process.env.REALESRGAN_MODEL || 'realesrgan-x4plus';
const FFMPEG_BIN = process.env.FFMPEG_BIN || 'ffmpeg';
const FFPROBE_BIN = process.env.FFPROBE_BIN || 'ffprobe';

async function makeTempDir(prefix) {
  const dir = path.join(os.tmpdir(), `${prefix}-${crypto.randomUUID()}`);
  await fs.mkdir(dir, { recursive: true });
  return dir;
}

async function getTelegramFileUrl(ctx, fileId) {
  const file = await ctx.api.getFile(fileId);
  return `https://api.telegram.org/file/bot${process.env.TELEGRAM_TOKEN}/${file.file_path}`;
}

async function downloadToFile(url, destPath) {
  const resp = await axios.get(url, { responseType: 'arraybuffer', timeout: 120000 });
  await fs.writeFile(destPath, Buffer.from(resp.data));
}

// Runs the ncnn-vulkan binary. Works on a single file or, given folder
// paths, on every image in a folder at once (used for video frames).
async function runRealEsrgan(inputPath, outputPath) {
  try {
    await execFileAsync(REALESRGAN_BIN, ['-i', inputPath, '-o', outputPath, '-n', REALESRGAN_MODEL], {
      maxBuffer: 1024 * 1024 * 32,
    });
  } catch (err) {
    if (err.code === 'ENOENT') {
      throw new Error(
        `Real-ESRGAN binary not found at ${REALESRGAN_BIN}. See the setup instructions in bot.js above enhanceImage/enhanceVideo.`
      );
    }
    throw new Error(err.stderr?.toString().trim() || err.message);
  }
}

async function enhanceImage(fileUrl) {
  const dir = await makeTempDir('enhance-img');
  try {
    const inputPath = path.join(dir, 'input.png');
    const outputPath = path.join(dir, 'output.png');
    await downloadToFile(fileUrl, inputPath);
    await runRealEsrgan(inputPath, outputPath);
    return await fs.readFile(outputPath);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
}

async function getVideoFps(inputPath) {
  const { stdout } = await execFileAsync(FFPROBE_BIN, [
    '-v', '0',
    '-select_streams', 'v:0',
    '-of', 'csv=p=0',
    '-show_entries', 'stream=r_frame_rate',
    inputPath,
  ]);
  const [num, den] = stdout.trim().split('/').map(Number);
  return den ? num / den : num || 30;
}

async function videoHasAudio(inputPath) {
  const { stdout } = await execFileAsync(FFPROBE_BIN, [
    '-v', '0',
    '-select_streams', 'a',
    '-show_entries', 'stream=index',
    '-of', 'csv=p=0',
    inputPath,
  ]);
  return stdout.trim().length > 0;
}

async function enhanceVideo(fileUrl) {
  const dir = await makeTempDir('enhance-vid');
  try {
    const inputPath = path.join(dir, 'input.mp4');
    const framesDir = path.join(dir, 'frames');
    const upscaledDir = path.join(dir, 'upscaled');
    const audioPath = path.join(dir, 'audio.aac');
    const outputPath = path.join(dir, 'output.mp4');
    await fs.mkdir(framesDir);
    await fs.mkdir(upscaledDir);

    await downloadToFile(fileUrl, inputPath);
    const fps = await getVideoFps(inputPath);
    const hasAudio = await videoHasAudio(inputPath);

    // 1. Extract frames.
    await execFileAsync(FFMPEG_BIN, ['-i', inputPath, path.join(framesDir, 'frame_%06d.png')]);

    // 2. Pull the audio track separately, if there is one.
    if (hasAudio) {
      await execFileAsync(FFMPEG_BIN, ['-i', inputPath, '-vn', '-acodec', 'aac', audioPath]);
    }

    // 3. Upscale every frame in one call (the binary accepts folder in/out).
    await runRealEsrgan(framesDir, upscaledDir);

    // 4. Reassemble frames (+ audio, if present) into the final video.
    const reassembleArgs = ['-r', String(fps), '-i', path.join(upscaledDir, 'frame_%06d.png')];
    if (hasAudio) reassembleArgs.push('-i', audioPath);
    reassembleArgs.push('-c:v', 'libx264', '-pix_fmt', 'yuv420p');
    if (hasAudio) reassembleArgs.push('-c:a', 'aac', '-shortest');
    reassembleArgs.push(outputPath);
    await execFileAsync(FFMPEG_BIN, reassembleArgs, { maxBuffer: 1024 * 1024 * 64 });

    return await fs.readFile(outputPath);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
}

// Keep a Telegram "typing/uploading" indicator alive during long-running
// generation calls (Telegram's chat action only lasts ~5s per call).
function keepChatActionAlive(ctx, action) {
  ctx.replyWithChatAction(action).catch(() => {});
  return setInterval(() => {
    ctx.replyWithChatAction(action).catch(() => {});
  }, 4000);
}

// /start
bot.command('start', async (ctx) => {
  userModes[ctx.chat.id] = null;
  await ctx.reply('👋 Hello! I\'m here. What can I do for you?', mainKeyboard);
});

// --- Mode switches ---
bot.hears('🤖 AI Chat', async (ctx) => {
  userModes[ctx.chat.id] = 'ai';
  await ctx.reply('🤖 AI Chat mode is ON.\n\nAsk me anything.');
});

bot.hears('🧠 GPT (AIML)', async (ctx) => {
  userModes[ctx.chat.id] = 'aiml';
  await ctx.reply('🧠 GPT (AIML) mode is ON.\n\nAsk me anything — this one runs through AIML API instead of Claude.');
});

bot.hears('🔎 Research', async (ctx) => {
  userModes[ctx.chat.id] = 'research';
  await ctx.reply('🔎 Research mode is ON (SerpAPI + Claude synthesis).\n\nWhat would you like me to research?');
});

bot.hears('🔵 Web', async (ctx) => {
  userModes[ctx.chat.id] = 'web';
  await ctx.reply('🔵 Web mode is ON (SerpAPI).\n\nWhat would you like me to search for?');
});

bot.hears('🎵 Music', async (ctx) => {
  userModes[ctx.chat.id] = 'music';
  await ctx.reply('🎵 Music mode is ON (free tier — MusicGen via Hugging Face).\n\nDescribe the track you want.');
});

bot.hears('📺 Videos', async (ctx) => {
  userModes[ctx.chat.id] = 'videos';
  await ctx.reply('📺 Video mode is ON (free tier — expect low quality and occasional failures).\n\nDescribe the video you want.');
});

bot.hears('🖼️ Images', async (ctx) => {
  userModes[ctx.chat.id] = 'images';
  await ctx.reply('🖼️ Image mode is ON (Pollinations — free).\n\nDescribe the image you want.');
});

bot.hears('✨ Enhance Image', async (ctx) => {
  userModes[ctx.chat.id] = 'enhance_image';
  await ctx.reply('✨ Image enhancement mode is ON (local Real-ESRGAN).\n\nSend me an image to enhance.');
});

bot.hears('📽️ Enhance Video', async (ctx) => {
  userModes[ctx.chat.id] = 'enhance_video';
  await ctx.reply('📽️ Video enhancement mode is ON (local Real-ESRGAN — can take a few minutes, longer with no GPU).\n\nSend me a video to enhance.');
});

// --- Actually handle what people type, based on mode ---
bot.on('message:text', async (ctx) => {
  const text = ctx.message.text;
  if (text.startsWith('/')) return; // let commands pass through
  // ignore taps on the menu buttons themselves (already handled above)
  const menuLabels = [
    '🤖 AI Chat', '🧠 GPT (AIML)', '🔎 Research', '🔵 Web', '🎵 Music',
    '📺 Videos', '🖼️ Images', '✨ Enhance Image', '📽️ Enhance Video',
  ];
  if (menuLabels.includes(text)) return;

  const mode = userModes[ctx.chat.id];

  if (mode === 'ai') {
    await ctx.replyWithChatAction('typing');
    try {
      const reply = await askClaude(text);
      await ctx.reply(reply || "I didn't get a text response back — try rephrasing.");
    } catch (err) {
      console.error('Anthropic API error:', err.response?.data || err.message);
      await ctx.reply('⚠️ Something went wrong calling the AI. Check the terminal for details.');
    }
    return;
  }

  if (mode === 'aiml') {
    await ctx.replyWithChatAction('typing');
    try {
      const response = await axios.post(
        'https://api.aimlapi.com/v1/chat/completions',
        {
          model: 'openai/gpt-5-5',
          messages: [{ role: 'user', content: text }],
        },
        {
          headers: {
            Authorization: `Bearer ${process.env.AIMLAPI_KEY}`,
            'Content-Type': 'application/json',
          },
        }
      );
      const reply = response.data.choices?.[0]?.message?.content;
      await ctx.reply(reply || "I didn't get a text response back — try rephrasing.");
    } catch (err) {
      console.error('AIML API error:', err.response?.data || err.message);
      await ctx.reply('⚠️ Something went wrong calling AIML API. Check the terminal for details.');
    }
    return;
  }

  if (mode === 'web') {
    await ctx.replyWithChatAction('typing');
    try {
      const results = await serpApiSearch(text, 6);
      await ctx.reply(formatSearchResults(results), { parse_mode: 'Markdown', disable_web_page_preview: true });
    } catch (err) {
      console.error('SerpAPI error:', err.response?.data || err.message);
      await ctx.reply('⚠️ Web search failed. Check the terminal for details.');
    }
    return;
  }

  if (mode === 'research') {
    await ctx.replyWithChatAction('typing');
    try {
      const answer = await researchTopic(text);
      await ctx.reply(answer, { parse_mode: 'Markdown', disable_web_page_preview: true });
    } catch (err) {
      console.error('Research error:', err.response?.data || err.message);
      await ctx.reply('⚠️ Research failed. Check the terminal for details.');
    }
    return;
  }

  if (mode === 'videos') {
    await ctx.reply('📺 Generating your video with a free Hugging Face model — this can take a while and sometimes fails under load...');
    const keepAlive = keepChatActionAlive(ctx, 'record_video');
    try {
      const videoBuffer = await generateVideo(text);
      await ctx.replyWithVideo(new InputFile(videoBuffer, 'video.mp4'), { caption: `📺 "${text}"` });
    } catch (err) {
      console.error('HF video generation error:', err.message);
      await ctx.reply(`⚠️ Free video generation failed: ${err.message}\nThis happens often on the free tier — try a shorter prompt or try again shortly.`);
    } finally {
      clearInterval(keepAlive);
    }
    return;
  }

  if (mode === 'music') {
    await ctx.reply('🎵 Generating your track with a free Hugging Face model (MusicGen)...');
    const keepAlive = keepChatActionAlive(ctx, 'record_voice');
    try {
      const audioBuffer = await generateMusic(text);
      await ctx.replyWithAudio(new InputFile(audioBuffer, 'music.flac'), { caption: `🎵 "${text}"` });
    } catch (err) {
      console.error('HF music generation error:', err.message);
      await ctx.reply(`⚠️ Free music generation failed: ${err.message}\nThis happens sometimes on the free tier — try again shortly.`);
    } finally {
      clearInterval(keepAlive);
    }
    return;
  }

  if (mode === 'images') {
    await ctx.reply('🖼️ Generating your image (Pollinations, free)...');
    const keepAlive = keepChatActionAlive(ctx, 'upload_photo');
    try {
      const imageBuffer = await generateImage(text);
      await ctx.replyWithPhoto(new InputFile(imageBuffer, 'image.png'), { caption: `🖼️ "${text}"` });
    } catch (err) {
      console.error('Pollinations image generation error:', err.response?.data || err.message);
      await ctx.reply('⚠️ Image generation failed. Check the terminal for details.');
    } finally {
      clearInterval(keepAlive);
    }
    return;
  }

  // Default / unknown text with no active mode
  await ctx.reply('Please choose an option from the menu below.', mainKeyboard);
});

// --- Enhance Image / Enhance Video (these arrive as photo/video uploads, not text) ---
bot.on('message:photo', async (ctx) => {
  if (userModes[ctx.chat.id] !== 'enhance_image') return;

  await ctx.reply('✨ Enhancing your image with Real-ESRGAN — this usually takes 10-30 seconds...');
  const keepAlive = keepChatActionAlive(ctx, 'upload_photo');
  try {
    const largest = ctx.message.photo[ctx.message.photo.length - 1];
    const fileUrl = await getTelegramFileUrl(ctx, largest.file_id);
    const enhancedBuffer = await enhanceImage(fileUrl);
    await ctx.replyWithPhoto(new InputFile(enhancedBuffer, 'enhanced.png'), { caption: '✨ Enhanced' });
  } catch (err) {
    console.error('Real-ESRGAN image enhancement error:', err.message);
    await ctx.reply(`⚠️ Image enhancement failed: ${err.message}`);
  } finally {
    clearInterval(keepAlive);
  }
});

bot.on('message:video', async (ctx) => {
  if (userModes[ctx.chat.id] !== 'enhance_video') return;

  await ctx.reply('📽️ Enhancing your video with Real-ESRGAN — this can take several minutes for longer clips...');
  const keepAlive = keepChatActionAlive(ctx, 'upload_video');
  try {
    const fileUrl = await getTelegramFileUrl(ctx, ctx.message.video.file_id);
    const enhancedBuffer = await enhanceVideo(fileUrl);
    await ctx.replyWithVideo(new InputFile(enhancedBuffer, 'enhanced.mp4'), { caption: '📽️ Enhanced' });
  } catch (err) {
    console.error('Real-ESRGAN video enhancement error:', err.message);
    await ctx.reply(`⚠️ Video enhancement failed: ${err.message}`);
  } finally {
    clearInterval(keepAlive);
  }
});

// Error handler
bot.catch((err) => {
  console.error('Bot error:', err);
});

bot.start();
console.log('🤖 Akram.ai is running...');
