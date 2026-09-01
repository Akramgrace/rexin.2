FROM node:20-slim

# ffmpeg is needed for video enhancement (frame extraction/reassembly).
# vulkan tools/libs let realesrgan-ncnn-vulkan run on CPU via software Vulkan
# when the host has no GPU (which is the case on Railway/Render).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libvulkan1 \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./
RUN npm install --omit=dev

COPY . .

# Make sure the Real-ESRGAN binary is executable inside the container
# (permissions from your machine don't always survive a git commit/zip).
RUN chmod +x ./bin/realesrgan-ncnn-vulkan || true

CMD ["node", "bot.js"]
