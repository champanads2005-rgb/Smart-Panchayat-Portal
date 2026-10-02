// Lets a citizen attach a photo of the civic issue and see the AI's
// image-based suggestion BEFORE submitting - calls
// /api/ai/analyze-image as an ephemeral preview (nothing is saved by
// this call). The citizen must click "Use this category" to apply the
// suggestion; nothing is auto-filled. If the model is unavailable, the
// image is invalid, or confidence is low, this shows that plainly and
// the form still works with manual category selection - a photo is
// never required to submit a complaint.
//
// Setting categoryEl.value here also dispatches a "change" event, which
// static/js/ai_priority.js already listens for - so accepting the
// image's category suggestion automatically feeds into the existing
// priority-prediction suggestion too, with no changes needed there.

(function () {
    const imageEl = document.getElementById("image");
    const categoryEl = document.getElementById("category");
    const resultEl = document.getElementById("ai-image-result");

    if (!imageEl || !categoryEl || !resultEl) return;

    imageEl.addEventListener("change", function () {
        resultEl.textContent = "";

        const file = imageEl.files[0];
        if (!file) return;

        resultEl.textContent = "Analyzing photo...";

        const formData = new FormData();
        formData.append("image", file);

        fetch("/api/ai/analyze-image", {
            method: "POST",
            body: formData,
        })
            .then((res) => res.json())
            .then((data) => {
                if (!data.available) {
                    const reason = data.reason || "unavailable";
                    if (reason === "file_too_large") {
                        resultEl.textContent =
                            "That photo is too large (max 5 MB). You can still submit without it, or try a smaller photo.";
                    } else if (reason && reason.startsWith("unsupported_format")) {
                        resultEl.textContent =
                            "That file isn't a supported image type (JPEG or PNG only). You can still submit without it.";
                    } else {
                        resultEl.textContent =
                            "AI image detection isn't available right now - you can still submit with a manually selected category.";
                    }
                    return;
                }

                const supported = (data.supported_classes || []).join(", ") || "none";

                if (data.detected_label === "Pothole" && !data.is_low_confidence) {
                    const confidencePct = Math.round(data.confidence * 100) + "%";
                    resultEl.innerHTML =
                        "AI detected: <strong>Pothole</strong> (confidence " + confidencePct + ") " +
                        "→ suggested category: <strong>" + data.suggested_category + "</strong> " +
                        '<button type="button" id="use-ai-image-category">Use this category</button>' +
                        '<div style="font-size:0.8em;color:#666;">Currently supports detecting: ' + supported +
                        ". Other issue types (garbage, drainage, street lights) are not yet supported by image detection - please select those manually.</div>";

                    document.getElementById("use-ai-image-category").addEventListener("click", function () {
                        categoryEl.value = data.suggested_category;
                        categoryEl.dispatchEvent(new Event("change"));
                    });
                } else if (data.detected_label === "Pothole" && data.is_low_confidence) {
                    const confidencePct = Math.round(data.confidence * 100) + "%";
                    resultEl.innerHTML =
                        "AI is uncertain about this photo (leaning pothole, confidence only " + confidencePct +
                        " - below the " + Math.round(data.confidence_threshold * 100) + "% threshold for a suggestion). " +
                        "Please select the category manually.";
                } else {
                    resultEl.innerHTML =
                        "AI did not detect a pothole in this photo. " +
                        '<div style="font-size:0.8em;color:#666;">Image detection currently only supports: ' + supported +
                        ". For other issue types, please select the category manually.</div>";
                }
            })
            .catch(function () {
                resultEl.textContent =
                    "Couldn't analyze the photo right now - you can still submit with a manually selected category.";
            });
    });
})();
