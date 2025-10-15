var bhvr_items = [
    {
        steps: [
            { video: "000003/Ground truth.mp4", label: "Ground truth" },
            { video: "000003/STeP.mp4", label: "Ours" },
            { video: "000003/IDM+BCS.mp4", label: "Baseline (image diffusion)" },
        ]
    },
    {
        steps: [
            { video: "000012/Ground truth.mp4", label: "Ground truth" },
            { video: "000012/STeP.mp4", label: "Ours" },
            { video: "000012/IDM+BCS.mp4", label: "Baseline (image diffusion)" },
        ]
    },
    {
        steps: [
            { video: "000046/Ground truth.mp4", label: "Ground truth" },
            { video: "000046/STeP.mp4", label: "Ours" },
            { video: "000046/IDM+BCS.mp4", label: "Baseline (image diffusion)" },
        ]
    },
    {
        steps: [
            { video: "000053/Ground truth.mp4", label: "Ground truth" },
            { video: "000053/STeP.mp4", label: "Ours" },
            { video: "000053/IDM+BCS.mp4", label: "Baseline (image diffusion)" },
        ]
    },
    {
        steps: [
            { video: "000067/Ground truth.mp4", label: "Ground truth" },
            { video: "000067/STeP.mp4", label: "Ours" },
            { video: "000067/IDM+BCS.mp4", label: "Baseline (image diffusion)" },
        ]
    },
    {
        steps: [
            { video: "000072/Ground truth.mp4", label: "Ground truth" },
            { video: "000072/STeP.mp4", label: "Ours" },
            { video: "000072/IDM+BCS.mp4", label: "Baseline (image diffusion)" },
        ]
    },
];

var dmri_items = [
    {
        steps: [
            { video: "000003/Ground truth.mp4", label: "Ground truth" },
            { video: "000003/STeP.mp4", label: "Ours" },
            { video: "000003/IDM+BCS.mp4", label: "Baseline (image diffusion)" },
        ]
    },
    {
        steps: [
            { video: "000016/Ground truth.mp4", label: "Ground truth" },
            { video: "000016/STeP.mp4", label: "Ours" },
            { video: "000016/IDM+BCS.mp4", label: "Baseline (image diffusion)" },
        ]
    },
    {
        steps: [
            { video: "000007/Ground truth.mp4", label: "Ground truth" },
            { video: "000007/STeP.mp4", label: "Ours" },
            { video: "000007/IDM+BCS.mp4", label: "Baseline (image diffusion)" },
        ]
    },
    {
        steps: [
            { video: "000011/Ground truth.mp4", label: "Ground truth" },
            { video: "000011/STeP.mp4", label: "Ours" },
            { video: "000011/IDM+BCS.mp4", label: "Baseline (image diffusion)" },
        ]
    },
    {
        steps: [
            { video: "000009/Ground truth.mp4", label: "Ground truth" },
            { video: "000009/STeP.mp4", label: "Ours" },
            { video: "000009/IDM+BCS.mp4", label: "Baseline (image diffusion)" },
        ]
    },
    {
        steps: [
            { video: "000013/Ground truth.mp4", label: "Ground truth" },
            { video: "000013/STeP.mp4", label: "Ours" },
            { video: "000013/IDM+BCS.mp4", label: "Baseline (image diffusion)" },
        ]
    },
];


function bhvr_carousel_item_template(item) {
    html = `<div class="x-card">
                <div class="x-labels">
                </div>
                <div class="x-row" style="flex-wrap: wrap;">`
    for (let i in item.steps) {
        let step = item.steps[i];
        html += `<div style="margin: 0 16px; flex: 1 0 120px; position: relative;">
                    <div class="x-labels">
                        <div class="x-label">${step.label}</div>
                    </div>
                    <div class="x-column">
                        <div style="width: 100%; aspect-ratio: 1; overflow: hidden; border-radius: 8px;">
                            <video autoplay playsinline loop muted height="100%" src="assets/bhvr-videos/${step.video}"></video>
                        </div>
                    </div>
                </div>`;
    }
    html += `</div></div>`;
    return html;
}


function dmri_carousel_item_template(item) {
    html = `<div class="x-card">
                <div class="x-labels">
                </div>
                <div class="x-row" style="flex-wrap: wrap;">`
    for (let i in item.steps) {
        let step = item.steps[i];
        html += `<div style="margin: 0 16px; flex: 1 0 120px; position: relative;">
                    <div class="x-labels">
                        <div class="x-label">${step.label}</div>
                    </div>
                    <div class="x-column">
                        <div style="width: 100%; aspect-ratio: 1; overflow: hidden; border-radius: 8px;">
                            <video autoplay playsinline loop muted height="100%" src="assets/dmri-videos/${step.video}"></video>
                        </div>
                    </div>
                </div>`;
    }
    html += `</div></div>`;
    return html;
}