import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

let scene, camera, renderer, controls, faceMesh;
let currentTexture = null;
let currentObjUrl = null;
const canvasContainer = document.getElementById('canvas-container');

// Download Logic
document.getElementById('download-obj').onclick = () => {
    if (!currentObjUrl) return;
    const a = document.createElement('a');
    a.href = currentObjUrl;
    a.download = 'face_reconstruction.obj';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
};

// Initialize 3D Engine
function initScene() {
    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 1);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
    canvasContainer.appendChild(renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    ambientLight.name = 'ambient';
    scene.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.4);
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);

    // Main light that follows the camera
    const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
    mainLight.position.set(1, 1, 1);
    mainLight.name = 'main_dir';
    camera.add(mainLight);
    scene.add(camera); // Camera must be in scene for its children to be rendered

    const fillLight = new THREE.PointLight(0x818cf8, 0.3);
    fillLight.position.set(-5, 3, 2);
    scene.add(fillLight);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Customization Event Listeners
    document.getElementById('mat-select').addEventListener('change', (e) => updateMaterial(e.target.value));
    document.getElementById('light-intensity').addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        scene.getObjectByName('main_dir').intensity = val;
        scene.getObjectByName('ambient').intensity = val * 0.4;
    });

    document.querySelectorAll('.color-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const color = e.target.dataset.color;
            document.querySelector('.color-btn.active')?.classList.remove('active');
            e.target.classList.add('active');
            // Animate background change if possible (basic for now)
            renderer.setClearColor(color, 0);
            canvasContainer.style.background = `radial-gradient(circle at center, ${color} 0%, #000 100%)`;
        });
    });

    window.addEventListener('resize', onWindowResize);
}

function updateMaterial(type) {
    if (!faceMesh) return;

    faceMesh.traverse(child => {
        if (child.isMesh) {
            let material;
            if (type === 'wireframe') {
                material = new THREE.MeshBasicMaterial({
                    color: 0x4f46e5,
                    wireframe: true,
                    name: 'wireframe'
                });
            } else { // Realistic
                material = new THREE.MeshStandardMaterial({
                    map: currentTexture,
                    color: currentTexture ? 0xffffff : 0xdbac98,
                    metalness: 0.05,
                    roughness: 0.5,
                    side: THREE.DoubleSide,
                    depthWrite: true,
                    transparent: false
                });
            }
            child.material = material;
        }
    });
}

function onWindowResize() {
    camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    renderer.render(scene, camera);
}

// Upload & Process Logic
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const uploadBtn = document.getElementById('upload-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const statusLabel = document.getElementById('status');

dropZone.onclick = () => fileInput.click();

fileInput.onchange = (e) => {
    if (e.target.files.length > 0) {
        processUpload(e.target.files[0]);
    }
};

let isPlaying = false;
let currentSequence = [];
let currentFrameIdx = 0;

// Video Control Listeners
const frameSlider = document.getElementById('frame-slider');
const videoControls = document.getElementById('video-controls');
const togglePlayBtn = document.getElementById('toggle-play');

frameSlider.oninput = (e) => {
    isPlaying = false; // Stop auto-play when user scrubs
    togglePlayBtn.textContent = "Chạy tiếp";
    currentFrameIdx = parseInt(e.target.value);
    const frameData = currentSequence[currentFrameIdx];
    if (frameData) {
        load3DFaceAsync(frameData.obj_url);
        updateFaceStats(frameData.face_status);
        currentObjUrl = frameData.obj_url;
    }
};

togglePlayBtn.onclick = () => {
    isPlaying = !isPlaying;
    togglePlayBtn.textContent = isPlaying ? "Tạm dừng" : "Chạy tiếp";
    if (isPlaying) playVideoSequence(currentSequence);
};

async function processUpload(file) {
    if (!file) return;

    loadingOverlay.classList.remove('hidden');
    videoControls.classList.add('hidden');
    isPlaying = false;
    firstFrameOffset = null;

    const isVideo = file.type.startsWith('video/');
    statusLabel.textContent = isVideo ? "Extracting frames & Processing video..." : "Detecting & Processing faces...";

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/reconstruct', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Failed to reconstruct');

        const data = await response.json();
        displayReconstruction(data);

    } catch (err) {
        console.error(err);
        statusLabel.textContent = "Error: " + err.message;
    } finally {
        loadingOverlay.classList.add('hidden');
    }
}

function displayReconstruction(data) {
    if (!data) return;

    if (data.type === 'video' && data.sequence) {
        currentSequence = data.sequence;
        statusLabel.textContent = `Video processed: ${data.sequence.length} frames`;
        
        // Setup Slider
        videoControls.classList.remove('hidden');
        frameSlider.max = data.sequence.length - 1;
        frameSlider.value = 0;

        if (data.tex_url) {
            const textureLoader = new THREE.TextureLoader();
            textureLoader.load(data.tex_url, (texture) => {
                texture.colorSpace = THREE.SRGBColorSpace;
                currentTexture = texture;
                togglePlayBtn.textContent = "Tạm dừng";
                
                // Show initial stats/URL
                if (data.sequence[0].face_status) updateFaceStats(data.sequence[0].face_status);
                currentObjUrl = data.sequence[0].obj_url;

                playVideoSequence(data.sequence);
            });
        } else {
            togglePlayBtn.textContent = "Tạm dừng";
            
            // Show initial stats/URL
            if (data.sequence[0].face_status) updateFaceStats(data.sequence[0].face_status);
            currentObjUrl = data.sequence[0].obj_url;

            playVideoSequence(data.sequence);
        }
    } else if (data.faces && data.faces.length > 0) {
        document.getElementById('face-list-container').classList.add('hidden');
        renderFaceList(data.faces);
        const firstFace = data.faces[0];
        load3DFace(firstFace.obj_url, firstFace.tex_url);
        updateFaceStats(firstFace.face_status);
        statusLabel.textContent = `Found ${data.faces.length} faces!`;
    }
}

async function playVideoSequence(sequence) {
    isPlaying = true;
    while (isPlaying) {
        for (let i = currentFrameIdx; i < sequence.length; i++) {
            if (!isPlaying) {
                currentFrameIdx = i; // Save progress
                return;
            }
            currentFrameIdx = i;
            frameSlider.value = i;
            const frameData = sequence[i];
            currentObjUrl = frameData.obj_url;
            updateFaceStats(frameData.face_status);
            await load3DFaceAsync(frameData.obj_url);
            await new Promise(r => setTimeout(r, 100));
        }
        currentFrameIdx = 0; // Reset for loop
    }
}

let firstFrameOffset = null;

function load3DFaceAsync(objUrl) {
    return new Promise((resolve) => {
        const loader = new OBJLoader();
        loader.load(objUrl, (object) => {
            if (faceMesh) scene.remove(faceMesh);
            const matType = document.getElementById('mat-select').value;
            faceMesh = object;

            faceMesh.traverse(child => {
                if (child.isMesh) {
                    child.geometry.computeVertexNormals();
                    if (matType === 'wireframe') {
                        child.material = new THREE.MeshBasicMaterial({
                            color: 0x4f46e5,
                            wireframe: true
                        });
                    } else {
                        child.material = new THREE.MeshStandardMaterial({
                            map: currentTexture,
                            color: currentTexture ? 0xffffff : 0xdbac98,
                            roughness: 0.5,
                            side: THREE.DoubleSide,
                            depthWrite: true,
                            transparent: false
                        });
                    }
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });

            // Stabilize position across frames
            if (firstFrameOffset === null) {
                const box = new THREE.Box3().setFromObject(object);
                firstFrameOffset = box.getCenter(new THREE.Vector3());
            }

            object.position.sub(firstFrameOffset);
            scene.add(object);
            resolve();
        });
    });
}

function renderFaceList(faces) {
    const listContainer = document.getElementById('face-list-container');
    const list = document.getElementById('face-list');

    list.innerHTML = '';
    listContainer.classList.remove('hidden');

    faces.forEach((face, idx) => {
        const item = document.createElement('div');
        item.className = `face-item ${idx === 0 ? 'active' : ''}`;
        item.textContent = idx + 1;
        item.onclick = () => {
            document.querySelector('.face-item.active')?.classList.remove('active');
            item.classList.add('active');
            load3DFace(face.obj_url, face.tex_url);
            updateFaceStats(face.face_status);
        };
        list.appendChild(item);
    });
}

function updateFaceStats(stats) {
    if (!stats) return;

    const statsCard = document.getElementById('face-stats');
    statsCard.classList.remove('hidden');

    const poseVal = document.getElementById('pose-val');
    const { pitch, yaw, roll } = stats.pose;
    poseVal.textContent = `P: ${pitch}°, Y: ${yaw}°, R: ${roll}°`;

    const mouthVal = document.getElementById('mouth-val');
    const mouthOpen = stats.expression.mouth_open;
    if (mouthOpen < 0.1) mouthVal.textContent = "Đang đóng";
    else if (mouthOpen < 0.5) mouthVal.textContent = "Hơi mở";
    else mouthVal.textContent = "Đang mở";

    const expBar = document.getElementById('exp-bar');
    const intensity = Math.min(stats.expression.intensity * 5, 100);
    expBar.style.width = `${intensity}%`;
}

function load3DFace(objUrl, texUrl) {
    // Clear previous face
    if (faceMesh) scene.remove(faceMesh);
    currentTexture = null;
    currentObjUrl = objUrl;

    const loader = new OBJLoader();

    const applyMaterial = (object, texture = null) => {
        currentTexture = texture;
        object.traverse((child) => {
            if (child.isMesh) {
                child.geometry.computeVertexNormals();
            }
        });

        // Use the current selection in the dropdown to set initial material
        const matType = document.getElementById('mat-select').value;
        faceMesh = object;
        updateMaterial(matType);

        // Center and scale
        const box = new THREE.Box3().setFromObject(object);
        const center = box.getCenter(new THREE.Vector3());
        object.position.sub(center);

        scene.add(object);
    };

    if (texUrl) {
        const textureLoader = new THREE.TextureLoader();
        textureLoader.load(texUrl, (texture) => {
            texture.colorSpace = THREE.SRGBColorSpace;
            loader.load(objUrl, (object) => {
                applyMaterial(object, texture);
                statusLabel.textContent = "Model Loaded with Texture";
            });
        });
    } else {
        loader.load(objUrl, (object) => {
            applyMaterial(object);
            statusLabel.textContent = "Model Loaded (No Texture)";
        });
    }
}

// Camera control
document.getElementById('reset-cam').onclick = () => {
    controls.reset();
    camera.position.set(0, 0, 3);
};

// Sidebar Resizer Logic
function initResizer() {
    const resizer = document.getElementById('resizer');
    const sidebar = document.getElementById('sidebar');
    const viewer = document.getElementById('viewer');
    let isResizing = false;
    let animationFrameId = null;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        resizer.classList.add('active');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        viewer.style.pointerEvents = 'none'; // Prevent Three.js interaction while dragging
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        if (animationFrameId) cancelAnimationFrame(animationFrameId);

        animationFrameId = requestAnimationFrame(() => {
            let newWidth = e.clientX;

            // Apply constraints
            if (newWidth < 250) newWidth = 250;
            if (newWidth > 600) newWidth = 600;

            sidebar.style.width = `${newWidth}px`;

            // Update Three.js viewport
            onWindowResize();
        });
    });

    document.addEventListener('mouseup', () => {
        if (!isResizing) return;
        isResizing = false;
        resizer.classList.remove('active');
        document.body.style.cursor = 'default';
        document.body.style.userSelect = 'auto';
        viewer.style.pointerEvents = 'auto';
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
    });
}

// Start
initScene();
initResizer();
animate();

// Drag and drop events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
    }, false);
});

dropZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    processUpload(file);
});
