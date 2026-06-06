import os
import base64
import cv2
import json
import streamlit.components.v1 as components

def image_to_base64(image_np):
    _, buffer = cv2.imencode('.jpg', image_np)
    return base64.b64encode(buffer).decode('utf-8')

# Create the component directory and index.html
component_dir = os.path.join(os.path.dirname(__file__), "cropper_component")
os.makedirs(component_dir, exist_ok=True)
html_path = os.path.join(component_dir, "index.html")

html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {
            margin: 0;
            padding: 0;
            background-color: #2E2E2E;
            display: flex;
            flex-direction: column;
            align-items: center;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: white;
            overflow: hidden;
        }
        .header {
            width: 100%;
            padding: 15px 20px;
            background: #1A1A1A;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-sizing: border-box;
            border-bottom: 1px solid #333;
        }
        .title {
            font-size: 16px;
            font-weight: 600;
            color: #EEE;
        }
        .btn {
            background: #C97D98;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
            box-shadow: 0 2px 8px rgba(201, 125, 152, 0.4);
        }
        .btn:hover { background: #A35271; }
        .btn:active { transform: scale(0.96); }
        .container {
            position: relative;
            margin-top: 20px;
            max-width: 90vw;
            max-height: 75vh;
        }
        img {
            display: block;
            max-width: 100%;
            max-height: 75vh;
            user-select: none;
            pointer-events: none;
            border-radius: 8px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }
        svg {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: visible;
        }
        .polygon {
            fill: rgba(201, 125, 152, 0.25);
            stroke: #C97D98;
            stroke-width: 3;
        }
        .handle {
            fill: white;
            stroke: #C97D98;
            stroke-width: 5;
            cursor: grab;
            transition: r 0.15s cubic-bezier(0.18, 0.89, 0.32, 1.28);
        }
        .handle:hover { r: 12; }
        .handle:active { cursor: grabbing; r: 10; }
        .magnifier {
            position: absolute;
            width: 120px;
            height: 120px;
            border: 3px solid #C97D98;
            border-radius: 50%;
            background-repeat: no-repeat;
            pointer-events: none;
            display: none;
            box-shadow: 0 8px 16px rgba(0,0,0,0.6);
            z-index: 10;
            background-color: #222;
        }
        .crosshair {
            position: absolute;
            top: 50%;
            left: 50%;
            width: 14px;
            height: 14px;
            transform: translate(-50%, -50%);
            pointer-events: none;
        }
        .crosshair::before, .crosshair::after {
            content: '';
            position: absolute;
            background: #C97D98;
            border-radius: 2px;
        }
        .crosshair::before { top: 6px; left: 0; width: 14px; height: 2px; }
        .crosshair::after { top: 0; left: 6px; width: 2px; height: 14px; }
        .hint {
            margin-top: 15px;
            margin-bottom: 20px;
            font-size: 13px;
            color: #AAA;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">✨ ลากจุด 4 มุมเพื่อครอบให้พอดีกับเอกสาร</div>
        <button class="btn" id="confirmBtn">ยืนยันการครอบตัด</button>
    </div>
    <div class="container" id="container">
        <img id="img" src="" />
        <svg id="svg">
            <polygon id="poly" class="polygon" points="" />
            <circle class="handle" id="p0" cx="0" cy="0" r="9" />
            <circle class="handle" id="p1" cx="0" cy="0" r="9" />
            <circle class="handle" id="p2" cx="0" cy="0" r="9" />
            <circle class="handle" id="p3" cx="0" cy="0" r="9" />
        </svg>
        <div class="magnifier" id="magnifier">
            <div class="crosshair"></div>
        </div>
    </div>
    <div class="hint">คุณสามารถลากจุดวงกลมทั้ง 4 มุม เพื่อปรับขอบกระดาษ</div>

    <script>
        function sendMessageToStreamlitClient(type, data) {
            const outData = Object.assign({
                isStreamlitMessage: true,
                type: type,
            }, data);
            window.parent.postMessage(outData, "*");
        }
        function init() { sendMessageToStreamlitClient("streamlit:componentReady", {apiVersion: 1}); }
        function setFrameHeight(height) { sendMessageToStreamlitClient("streamlit:setFrameHeight", {height: height}); }
        function sendValue(value) { sendMessageToStreamlitClient("streamlit:setComponentValue", {value: value}); }

        let imgW = 0;
        let imgH = 0;
        let initPts = [];
        let points = [];
        let displayW = 0;
        let displayH = 0;
        let scaleX = 1;
        let scaleY = 1;
        let draggingIdx = -1;
        let hasInitialized = false;

        const container = document.getElementById('container');
        const img = document.getElementById('img');
        const svg = document.getElementById('svg');
        const poly = document.getElementById('poly');
        const mag = document.getElementById('magnifier');
        const handles = [
            document.getElementById('p0'),
            document.getElementById('p1'),
            document.getElementById('p2'),
            document.getElementById('p3')
        ];

        function draw() {
            poly.setAttribute('points', points.map(p => `${p.x},${p.y}`).join(' '));
            handles.forEach((h, i) => {
                h.setAttribute('cx', points[i].x);
                h.setAttribute('cy', points[i].y);
            });
        }

        function updateGeometry() {
            if(!img.complete || img.naturalWidth === 0) return;
            const rect = img.getBoundingClientRect();
            displayW = rect.width;
            displayH = rect.height;
            scaleX = displayW / imgW;
            scaleY = displayH / imgH;
            
            points = initPts.map(p => ({ x: p[0] * scaleX, y: p[1] * scaleY }));
            draw();
            setTimeout(() => setFrameHeight(document.body.scrollHeight), 100);
        }

        img.onload = updateGeometry;
        window.onresize = updateGeometry;

        window.addEventListener("message", function(event) {
            if (event.data.type === "streamlit:render") {
                const args = event.data.args;
                if (!hasInitialized) {
                    img.src = "data:image/jpeg;base64," + args.img_b64;
                    imgW = args.w;
                    imgH = args.h;
                    initPts = args.pts;
                    hasInitialized = true;
                    // Trigger manual geometry update if already loaded
                    if (img.complete) updateGeometry();
                }
            }
        });

        // --- Dragging Logic ---
        function getMousePos(evt) {
            const rect = svg.getBoundingClientRect();
            return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
        }

        function updateMagnifier(x, y) {
            mag.style.display = 'block';
            let magLeft = x - 140;
            let magTop = y - 140;
            if (magLeft < 0) magLeft = x + 40;
            if (magTop < 0) magTop = y + 40;
            mag.style.left = magLeft + 'px';
            mag.style.top = magTop + 'px';
            
            const ratio = 2.5;
            mag.style.backgroundImage = `url('${img.src}')`;
            mag.style.backgroundSize = `${displayW * ratio}px ${displayH * ratio}px`;
            const bgX = -(x * ratio - 60);
            const bgY = -(y * ratio - 60);
            mag.style.backgroundPosition = `${bgX}px ${bgY}px`;
        }

        svg.addEventListener('mousedown', (e) => {
            const pos = getMousePos(e);
            let minDist = 25;
            handles.forEach((h, i) => {
                const dx = points[i].x - pos.x;
                const dy = points[i].y - pos.y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < minDist) { minDist = dist; draggingIdx = i; }
            });
            if (draggingIdx !== -1) updateMagnifier(pos.x, pos.y);
        });

        svg.addEventListener('mousemove', (e) => {
            if (draggingIdx === -1) return;
            let pos = getMousePos(e);
            pos.x = Math.max(0, Math.min(displayW, pos.x));
            pos.y = Math.max(0, Math.min(displayH, pos.y));
            points[draggingIdx] = pos;
            draw();
            updateMagnifier(pos.x, pos.y);
        });

        const endDrag = () => { draggingIdx = -1; mag.style.display = 'none'; };
        svg.addEventListener('mouseup', endDrag);
        svg.addEventListener('mouseleave', endDrag);

        svg.addEventListener('touchstart', (e) => {
            const touch = e.touches[0];
            const pos = getMousePos(touch);
            let minDist = 40;
            handles.forEach((h, i) => {
                const dx = points[i].x - pos.x;
                const dy = points[i].y - pos.y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < minDist) { minDist = dist; draggingIdx = i; }
            });
            if (draggingIdx !== -1) { e.preventDefault(); updateMagnifier(pos.x, pos.y); }
        }, {passive: false});

        svg.addEventListener('touchmove', (e) => {
            if (draggingIdx === -1) return;
            e.preventDefault();
            const touch = e.touches[0];
            let pos = getMousePos(touch);
            pos.x = Math.max(0, Math.min(displayW, pos.x));
            pos.y = Math.max(0, Math.min(displayH, pos.y));
            points[draggingIdx] = pos;
            draw();
            updateMagnifier(pos.x, pos.y);
        }, {passive: false});

        svg.addEventListener('touchend', endDrag);
        svg.addEventListener('touchcancel', endDrag);

        document.getElementById('confirmBtn').addEventListener('click', () => {
            document.getElementById('confirmBtn').innerText = 'กำลังบันทึก...';
            document.getElementById('confirmBtn').style.opacity = '0.7';
            
            const finalPts = points.map(p => [
                Math.round(p.x / scaleX),
                Math.round(p.y / scaleY)
            ]);
            
            sendValue(finalPts);
        });

        init();
    </script>
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

_cropper_component = components.declare_component("cropper_component", path=component_dir)

def st_cropper(image_np, pts, key=None):
    """
    Renders the interactive cropper as a proper Streamlit component.
    Returns the final coordinates (list of 4 [x, y] points) when the user confirms.
    """
    img_b64 = image_to_base64(image_np)
    h, w = image_np.shape[:2]
    
    if pts is None or len(pts) != 4:
        pts = [[0, 0], [w, 0], [w, h], [0, h]]
    elif hasattr(pts, 'tolist'):
        pts = pts.tolist()
        
    return _cropper_component(img_b64=img_b64, w=w, h=h, pts=pts, key=key, default=None)
