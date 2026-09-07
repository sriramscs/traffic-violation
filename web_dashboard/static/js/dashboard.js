/**
 * AURA-TRAFFIC AI Dashboard Client Logic
 * Real-time stats polling, evidence rendering, Chart.js updates, and modal inspector.
 */

let pieChart = null;
let barChart = null;
let currentViolationsData = [];

document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    fetchStats();
    loadViolations();
    loadBenchmarkTable();

    // Periodic live polling
    setInterval(fetchStats, 1500);
    setInterval(loadViolations, 3000);
});

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));

    const targetTab = document.getElementById(`tab-${tabId}`);
    if (targetTab) targetTab.classList.add("active");

    const clickedBtn = Array.from(document.querySelectorAll(".tab-btn")).find(btn => 
        btn.getAttribute("onclick").includes(tabId)
    );
    if (clickedBtn) clickedBtn.classList.add("active");

    if (tabId === "analytics-view" && pieChart && barChart) {
        pieChart.update();
        barChart.update();
    }
}

// Fetch Real-Time KPIs & Stats
async function fetchStats() {
    try {
        const res = await fetch("/api/stats");
        const data = await res.json();

        // Update KPIs
        animateCounter("kpi-total", data.counts.total);
        animateCounter("kpi-helmet", data.counts.helmet);
        animateCounter("kpi-triple", data.counts.triple);
        animateCounter("kpi-seatbelt", data.counts.seatbelt);
        animateCounter("kpi-phone", data.counts.phone);

        // Update Active Tracks Badge
        const tracksBadge = document.getElementById("badge-active-tracks");
        if (tracksBadge) tracksBadge.textContent = `${data.active_tracks} Active Tracks`;

        // Update ViT Button State
        const vitBtn = document.getElementById("vit-toggle-btn");
        const vitLabel = document.getElementById("vit-btn-label");
        if (data.use_vit) {
            vitBtn.classList.add("active");
            vitLabel.textContent = "ViT Enhancement: ON";
        } else {
            vitBtn.classList.remove("active");
            vitLabel.textContent = "ViT Enhancement: OFF (Baseline)";
        }

        // Update GPU Button State
        const gpuBtn = document.getElementById("gpu-toggle-btn");
        const gpuLabel = document.getElementById("gpu-btn-label");
        if (gpuBtn && gpuLabel) {
            if (data.device && !data.device.includes("CPU (GPU Disabled)")) {
                gpuBtn.classList.add("active");
                gpuBtn.classList.remove("off");
                const shortDevice = data.device.includes("RTX") ? "RTX 3050" : data.device;
                gpuLabel.textContent = `GPU: ${shortDevice} (ON)`;
            } else {
                gpuBtn.classList.remove("active");
                gpuBtn.classList.add("off");
                gpuLabel.textContent = "Hardware: CPU Only";
            }
        }

        // Update Charts
        updateChartsData(data.counts);

    } catch (err) {
        console.error("Failed to fetch stats:", err);
    }
}

function animateCounter(id, targetVal) {
    const el = document.getElementById(id);
    if (!el) return;
    const current = parseInt(el.textContent) || 0;
    if (current !== targetVal) {
        el.textContent = targetVal;
    }
}

// Fetch and Render Violations Log & Evidence
async function loadViolations() {
    try {
        const res = await fetch("/api/violations");
        const violations = await res.json();
        currentViolationsData = violations;

        renderLiveFeed(violations);
        renderEvidenceGallery(violations);

    } catch (err) {
        console.error("Failed to load violations:", err);
    }
}

// Render Right-Hand Activity Stream
function renderLiveFeed(violations) {
    const feedContainer = document.getElementById("feed-container");
    if (!feedContainer) return;

    if (!violations || violations.length === 0) {
        feedContainer.innerHTML = `
            <div class="feed-empty-state">
                <span class="empty-icon">⏳</span>
                <p>Monitoring traffic stream for violations...</p>
            </div>
        `;
        return;
    }

    const recent = violations.slice(0, 15);
    feedContainer.innerHTML = recent.map(v => `
        <div class="feed-item" onclick="openEvidenceModal('${v.id}')">
            <div class="feed-item-top">
                <span class="feed-badge badge-danger">${v.violation_type.replace('_', ' ')}</span>
                <span class="feed-timestamp">${v.timestamp.split(' ')[1]}</span>
            </div>
            <div class="feed-plate">
                <span>🚗</span>
                <span>${v.plate_number || 'UNKNOWN PLATE'}</span>
            </div>
            <div class="feed-footer">
                <span>Track #${v.track_id} | Conf: ${v.confidence}%</span>
                <button class="xai-btn">View Grad-CAM++</button>
            </div>
        </div>
    `).join('');
}

// Render Gallery Tab
function renderEvidenceGallery(violations) {
    const galleryGrid = document.getElementById("evidence-gallery-grid");
    if (!galleryGrid) return;

    if (!violations || violations.length === 0) {
        galleryGrid.innerHTML = `
            <div class="feed-empty-state" style="grid-column: 1 / -1;">
                <span class="empty-icon">📂</span>
                <p>No violation evidence records recorded yet.</p>
            </div>
        `;
        return;
    }

    galleryGrid.innerHTML = violations.map(v => `
        <div class="evidence-card" onclick="openEvidenceModal('${v.id}')">
            <div class="evidence-img-wrap">
                <img src="${v.evidence_image}" alt="${v.violation_type}" loading="lazy">
            </div>
            <div class="evidence-body">
                <div class="feed-item-top">
                    <span class="feed-badge badge-danger">${v.violation_type.replace('_', ' ')}</span>
                    <span class="feed-timestamp">${v.timestamp}</span>
                </div>
                <div class="feed-plate">
                    <span>Plate: <strong>${v.plate_number || 'N/A'}</strong></span>
                </div>
                <div class="feed-footer">
                    <span>Vehicle #${v.track_id}</span>
                    <span style="color: var(--color-cyan);">Conf: ${v.confidence}%</span>
                </div>
            </div>
        </div>
    `).join('');
}

// Modal Inspector
function openEvidenceModal(violationId) {
    const v = currentViolationsData.find(item => item.id === violationId);
    if (!v) return;

    const modal = document.getElementById("evidence-modal");
    document.getElementById("modal-title").textContent = `Violation Dossier: ${v.violation_type.replace('_', ' ')} (#${v.track_id})`;
    document.getElementById("modal-img").src = v.evidence_image;
    document.getElementById("modal-download-btn").href = v.evidence_image;

    const metaGrid = document.getElementById("modal-meta");
    metaGrid.innerHTML = `
        <div class="meta-box">
            <span class="meta-label">Violation Type</span>
            <span class="meta-val" style="color: var(--color-red);">${v.violation_type.replace('_', ' ')}</span>
        </div>
        <div class="meta-box">
            <span class="meta-label">License Plate</span>
            <span class="meta-val">${v.plate_number || 'N/A'}</span>
        </div>
        <div class="meta-box">
            <span class="meta-label">Track ID</span>
            <span class="meta-val">#${v.track_id}</span>
        </div>
        <div class="meta-box">
            <span class="meta-label">Confidence</span>
            <span class="meta-val">${v.confidence}%</span>
        </div>
        <div class="meta-box">
            <span class="meta-label">Recorded Timestamp</span>
            <span class="meta-val">${v.timestamp}</span>
        </div>
        <div class="meta-box">
            <span class="meta-label">Explainable AI</span>
            <span class="meta-val" style="color: var(--color-cyan);">Grad-CAM++ Verified</span>
        </div>
    `;

    modal.classList.add("active");
}

function closeModal(event) {
    if (event && event.target !== event.currentTarget) return;
    document.getElementById("evidence-modal").classList.remove("active");
}

// Toggle ViT Enhancement
async function toggleViT() {
    try {
        const res = await fetch("/api/toggle_vit", { method: "POST" });
        const data = await res.json();
        const vitBtn = document.getElementById("vit-toggle-btn");
        const vitLabel = document.getElementById("vit-btn-label");

        if (data.use_vit) {
            vitBtn.classList.add("active");
            vitLabel.textContent = "ViT Enhancement: ON";
        } else {
            vitBtn.classList.remove("active");
            vitLabel.textContent = "ViT Enhancement: OFF (Baseline)";
        }
    } catch (err) {
        console.error("Failed to toggle ViT:", err);
    }
}

// Toggle GPU / CPU Execution Mode
async function toggleGPU() {
    const gpuBtn = document.getElementById("gpu-toggle-btn");
    const gpuLabel = document.getElementById("gpu-btn-label");
    if (gpuLabel) gpuLabel.textContent = "Switching Hardware...";

    try {
        const res = await fetch("/api/toggle_gpu", { method: "POST" });
        const data = await res.json();

        if (!data.success) {
            alert(data.error || "Could not toggle GPU");
            return;
        }

        if (data.force_cpu) {
            gpuBtn.classList.remove("active");
            gpuBtn.classList.add("off");
            gpuLabel.textContent = "Hardware: CPU Only";
        } else {
            gpuBtn.classList.add("active");
            gpuBtn.classList.remove("off");
            const shortDevice = data.device_name.includes("RTX") ? "RTX 3050" : data.device_name;
            gpuLabel.textContent = `GPU: ${shortDevice} (ON)`;
        }
    } catch (err) {
        console.error("Failed to toggle GPU:", err);
        if (gpuLabel) gpuLabel.textContent = "Toggle Error";
    }
}

// Switch Video Source
async function changeVideoSource(sourcePath) {
    if (sourcePath === "custom_url") {
        const customInput = prompt("Enter Video File Path (e.g. input/sample1.mp4) or RTSP/HTTP Stream URL:");
        if (!customInput) {
            document.getElementById("video-source-select").value = "input/sample3.mp4";
            return;
        }
        sourcePath = customInput.trim();
    }

    try {
        const res = await fetch("/api/switch_source", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: sourcePath })
        });
        const data = await res.json();
        if (data.success) {
            console.log("Switched video source to:", sourcePath);
            const camBadge = document.getElementById("camera-badge-label");
            if (camBadge && data.camera_id) {
                camBadge.textContent = "📹 " + data.camera_id;
            }
        } else {
            alert(data.error || "Failed to switch video source");
        }
    } catch (err) {
        console.error("Failed to switch source:", err);
    }
}

// Initialize Charts
function initCharts() {
    const pieCtx = document.getElementById("violationPieChart")?.getContext("2d");
    if (pieCtx) {
        pieChart = new Chart(pieCtx, {
            type: "doughnut",
            data: {
                labels: ["Helmet", "Triple Riding", "Seatbelt", "Phone", "Lane Change"],
                datasets: [{
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: [
                        "#ffa502",
                        "#ff4757",
                        "#3b82f6",
                        "#9b59b6",
                        "#00e5ff"
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "right",
                        labels: { color: "#94a3b8", font: { family: "Inter", size: 12 } }
                    }
                }
            }
        });
    }

    const barCtx = document.getElementById("violationBarChart")?.getContext("2d");
    if (barCtx) {
        barChart = new Chart(barCtx, {
            type: "bar",
            data: {
                labels: ["Helmet", "Triple", "Seatbelt", "Phone", "Lane"],
                datasets: [{
                    label: "Violations Recorded",
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: "#00e5ff",
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
                    y: { ticks: { color: "#94a3b8", stepSize: 1 }, grid: { color: "rgba(255,255,255,0.05)" } }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
}

function updateChartsData(counts) {
    if (pieChart) {
        pieChart.data.datasets[0].data = [
            counts.helmet || 0,
            counts.triple || 0,
            counts.seatbelt || 0,
            counts.phone || 0,
            counts.lane || 0
        ];
        pieChart.update();
    }

    if (barChart) {
        barChart.data.datasets[0].data = [
            counts.helmet || 0,
            counts.triple || 0,
            counts.seatbelt || 0,
            counts.phone || 0,
            counts.lane || 0
        ];
        barChart.update();
    }
}

// Load Benchmark Data for Tab 4
async function loadBenchmarkTable() {
    const tbody = document.getElementById("benchmark-tbody");
    if (!tbody) return;

    try {
        const res = await fetch("/api/benchmark_data");
        const rows = await res.json();

        tbody.innerHTML = rows.map(r => `
            <tr>
                <td><strong>${r.Architecture}</strong></td>
                <td>
                    <span class="badge-pill" style="background: ${r.ViT_Enabled ? 'rgba(46, 213, 115, 0.15)' : 'rgba(255, 71, 87, 0.15)'}; color: ${r.ViT_Enabled ? '#2ed573' : '#ff4757'};">
                        ${r.ViT_Enabled ? 'Active (ViT-B/16)' : 'None (Baseline)'}
                    </span>
                </td>
                <td>${r.Precision}</td>
                <td>${r.Recall}</td>
                <td><strong>${r.F1_Score}</strong></td>
                <td>${r.mAP_05 || r["mAP@0.5"]}</td>
                <td style="color: var(--color-cyan); font-weight: 600;">${r.FPS} FPS</td>
                <td>${r.Avg_Latency_ms}</td>
            </tr>
        `).join('');

    } catch (err) {
        console.error("Failed to load benchmark:", err);
    }
}
