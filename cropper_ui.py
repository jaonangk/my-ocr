import base64
import cv2
import json

def image_to_base64(image_np):
    _, buffer = cv2.imencode('.jpg', image_np)
    return base64.b64encode(buffer).decode('utf-8')

def get_cropper_html(image_np, pts):
    """
    Generate an HTML/JS snippet for 4-point interactive document cropping.
    """
    img_b64 = image_to_base64(image_np)
    h, w = image_np.shape[:2]
    
    # Default to corners if no pts
    if pts is None or len(pts) != 4:
        pts = [[0, 0], [w, 0], [w, h], [0, h]]
    else:
        # pts comes in as numpy array or list
        if hasattr(pts, 'tolist'):
            pts = pts.tolist()
            
    pts_json = json.dumps(pts)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #2E2E2E;
                display: flex;
                flex-direction: column;
                align-items: center;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                color: white;
            }}
            .header {{
                width: 100%;
                padding: 15px 20px;
                background: #1A1A1A;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-sizing: border-box;
                border-bottom: 1px solid #333;
            }}
            .title {{
                font-size: 16px;
                font-weight: 600;
                color: #EEE;
            }}
            .btn {{
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
            }}
            .btn:hover {{
                background: #A35271;
            }}
            .btn:active {{
                transform: scale(0.96);
            }}
            .container {{
                position: relative;
                margin-top: 20px;
                max-width: 90vw;
                max-height: 75vh;
            }}
            img {{
                display: block;
                max-width: 100%;
                max-height: 75vh;
                user-select: none;
                pointer-events: none;
                border-radius: 8px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.5);
            }}
            svg {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                overflow: visible;
            }}
            .polygon {{
                fill: rgba(201, 125, 152, 0.25);
                stroke: #C97D98;
                stroke-width: 3;
            }}
            .handle {{
                fill: white;
                stroke: #C97D98;
                stroke-width: 5;
                cursor: grab;
                transition: r 0.15s cubic-bezier(0.18, 0.89, 0.32, 1.28);
            }}
            .handle:hover {{
                r: 12;
            }}
            .handle:active {{
                cursor: grabbing;
                r: 10;
            }}
            .magnifier {{
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
            }}
            .crosshair {{
                position: absolute;
                top: 50%;
                left: 50%;
                width: 14px;
                height: 14px;
                transform: translate(-50%, -50%);
                pointer-events: none;
            }}
            .crosshair::before, .crosshair::after {{
                content: '';
                position: absolute;
                background: #C97D98;
                border-radius: 2px;
            }}
            .crosshair::before {{ top: 6px; left: 0; width: 14px; height: 2px; }}
            .crosshair::after {{ top: 0; left: 6px; width: 2px; height: 14px; }}
            .hint {{
                margin-top: 15px;
                font-size: 13px;
                color: #AAA;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="title">✨ ลากจุด 4 มุมเพื่อครอบให้พอดีกับเอกสาร</div>
            <button class="btn" id="confirmBtn">ยืนยันการครอบตัด</button>
        </div>
        <div class="container" id="container">
            <img id="img" src="data:image/jpeg;base64,{img_b64}" />
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
        <div class="hint">คุณสามารถลากจุดวงกลมทั้ง 4 มุม เพื่อปรับขอบกระดาษให้ตรงเป๊ะได้เหมือนแอปสแกนเนอร์มือถือ</div>

        <script>
            const imgW = {w};
            const imgH = {h};
            let initPts = {pts_json};
            
            // Reorder points to ensure TL, TR, BR, BL order visually if possible
            // We trust the backend provided a reasonable polygon, but let's make sure it's usable.
            
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

            let displayW = 0;
            let displayH = 0;
            let scaleX = 1;
            let scaleY = 1;
            let points = [];

            function updateGeometry() {{
                const rect = img.getBoundingClientRect();
                displayW = rect.width;
                displayH = rect.height;
                scaleX = displayW / imgW;
                scaleY = displayH / imgH;
                
                points = initPts.map(p => ({{ x: p[0] * scaleX, y: p[1] * scaleY }}));
                draw();
            }}

            function draw() {{
                poly.setAttribute('points', points.map(p => `${{p.x}},${{p.y}}`).join(' '));
                handles.forEach((h, i) => {{
                    h.setAttribute('cx', points[i].x);
                    h.setAttribute('cy', points[i].y);
                }});
            }}

            // Important: wait for image to decode layout before reading dimensions
            if (img.complete) {{
                updateGeometry();
            }} else {{
                img.onload = updateGeometry;
            }}
            window.onresize = updateGeometry;

            // Dragging logic
            let draggingIdx = -1;

            function getMousePos(evt) {{
                const rect = svg.getBoundingClientRect();
                return {{
                    x: evt.clientX - rect.left,
                    y: evt.clientY - rect.top
                }};
            }}

            function updateMagnifier(x, y) {{
                mag.style.display = 'block';
                // Position magnifier slightly offset from cursor (avoid off-screen)
                let magLeft = x - 140;
                let magTop = y - 140;
                if (magLeft < 0) magLeft = x + 40;
                if (magTop < 0) magTop = y + 40;
                
                mag.style.left = magLeft + 'px';
                mag.style.top = magTop + 'px';
                
                const ratio = 2.5; // Zoom level
                mag.style.backgroundImage = `url('${{img.src}}')`;
                mag.style.backgroundSize = `${{displayW * ratio}}px ${{displayH * ratio}}px`;
                
                const bgX = -(x * ratio - 60);
                const bgY = -(y * ratio - 60);
                mag.style.backgroundPosition = `${{bgX}}px ${{bgY}}px`;
            }}

            svg.addEventListener('mousedown', (e) => {{
                const pos = getMousePos(e);
                let minDist = 25; // grab radius
                handles.forEach((h, i) => {{
                    const dx = points[i].x - pos.x;
                    const dy = points[i].y - pos.y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist < minDist) {{
                        minDist = dist;
                        draggingIdx = i;
                    }}
                }});
                if (draggingIdx !== -1) {{
                    updateMagnifier(pos.x, pos.y);
                }}
            }});

            svg.addEventListener('mousemove', (e) => {{
                if (draggingIdx === -1) return;
                let pos = getMousePos(e);
                pos.x = Math.max(0, Math.min(displayW, pos.x));
                pos.y = Math.max(0, Math.min(displayH, pos.y));
                points[draggingIdx] = pos;
                draw();
                updateMagnifier(pos.x, pos.y);
            }});

            const endDrag = () => {{
                draggingIdx = -1;
                mag.style.display = 'none';
            }};

            svg.addEventListener('mouseup', endDrag);
            svg.addEventListener('mouseleave', endDrag);

            // Touch support for mobile
            svg.addEventListener('touchstart', (e) => {{
                const touch = e.touches[0];
                const pos = getMousePos(touch);
                let minDist = 40; // larger grab radius for touch
                handles.forEach((h, i) => {{
                    const dx = points[i].x - pos.x;
                    const dy = points[i].y - pos.y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist < minDist) {{
                        minDist = dist;
                        draggingIdx = i;
                    }}
                }});
                if (draggingIdx !== -1) {{
                    e.preventDefault();
                    updateMagnifier(pos.x, pos.y);
                }}
            }}, {{passive: false}});

            svg.addEventListener('touchmove', (e) => {{
                if (draggingIdx === -1) return;
                e.preventDefault();
                const touch = e.touches[0];
                let pos = getMousePos(touch);
                pos.x = Math.max(0, Math.min(displayW, pos.x));
                pos.y = Math.max(0, Math.min(displayH, pos.y));
                points[draggingIdx] = pos;
                draw();
                updateMagnifier(pos.x, pos.y);
            }}, {{passive: false}});

            svg.addEventListener('touchend', endDrag);
            svg.addEventListener('touchcancel', endDrag);

            // Confirm Action
            document.getElementById('confirmBtn').addEventListener('click', () => {{
                document.getElementById('confirmBtn').innerText = 'กำลังบันทึก...';
                document.getElementById('confirmBtn').style.opacity = '0.7';
                
                const finalPts = points.map(p => [
                    Math.round(p.x / scaleX),
                    Math.round(p.y / scaleY)
                ]);
                
                try {{
                    const url = new URL(window.parent.location.href);
                    url.searchParams.set("triggered_event", "crop_confirmed");
                    url.searchParams.set("crop_pts", JSON.stringify(finalPts));
                    window.parent.location.href = url.toString();
                }} catch(err) {{
                    // Fallback to postMessage if cross-origin policy prevents direct access
                    window.parent.postMessage({{
                        type: 'recalpt_action', 
                        value: 'crop_confirmed',
                        points: finalPts
                    }}, '*');
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html
