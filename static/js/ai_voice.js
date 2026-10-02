// Lets a citizen record (via MediaRecorder) or upload an audio file,
// see a REAL transcript from /api/ai/transcribe-audio (ephemeral - not
// saved by this call), and explicitly accept it into the existing
// #description textarea for review/editing before submitting. Nothing
// is ever auto-submitted: accepting a transcript only fills the text
// box, and the citizen still has to review it and click "Submit
// Complaint" themselves, and can edit the text freely first.
//
// Because this reuses the EXISTING #description textarea, accepting a
// transcript automatically re-triggers ai_classify.js's "input" listener
// (Phase 2) with no changes needed there - classification, priority
// suggestion, and (server-side, at review time) duplicate detection all
// key off complaints.description regardless of whether it was typed or
// transcribed.

(function () {
    const audioInput = document.getElementById("audio");
    const recordBtn = document.getElementById("record-btn");
    const stopBtn = document.getElementById("stop-btn");
    const previewEl = document.getElementById("audio-preview");
    const resultEl = document.getElementById("ai-voice-result");
    const descriptionEl = document.getElementById("description");

    if (!audioInput || !resultEl || !descriptionEl) return;

    let mediaRecorder = null;
    let recordedChunks = [];

    function getSelectedLanguage() {
        const checked = document.querySelector('input[name="audio_language"]:checked');
        return checked ? checked.value : "en";
    }

    function putFileIntoInput(blob, filename) {
        const file = new File([blob], filename, { type: blob.type });
        const dt = new DataTransfer();
        dt.items.add(file);
        audioInput.files = dt.files;
    }

    function showPreview(blob) {
        previewEl.src = URL.createObjectURL(blob);
        previewEl.style.display = "block";
    }

    function renderResult(data) {
        if (!data.available) {
            const reason = data.reason || "unavailable";
            const messages = {
                file_too_large: "That recording is too large (max 10 MB).",
                audio_too_short: "That recording is too short - please record at least a second of speech.",
                not_a_valid_audio_file: "That file isn't a supported audio format.",
                audio_appears_silent: "No speech was detected in that recording (it appears silent) - please try again, speaking clearly into the microphone.",
                unsupported_language: "That language isn't supported for voice complaints yet.",
                model_download_blocked: "Kannada speech-to-text isn't available in this deployment (no reachable speech model). You can still type your complaint in Kannada or English below.",
            };
            let message = messages[reason];
            if (!message && reason && reason.startsWith("audio_too_long")) {
                message = "That recording is too long (max 2 minutes).";
            }
            resultEl.innerHTML = '<span style="color:#a33;">' + (message || "Couldn't transcribe that recording - you can still type your complaint below.") + "</span>";
            return;
        }

        const confidencePct = Math.round(data.confidence * 100) + "%";
        const engine = data.engine || "speech recognizer";
        const uncertainNote = data.is_low_confidence
            ? ' <span style="color:#a33;">(low confidence - please read this carefully)</span>'
            : "";

        if (!data.transcript) {
            resultEl.innerHTML =
                'No words were recognized in that recording. You can try again or type your complaint directly below.';
            return;
        }

        resultEl.innerHTML =
            '<div style="border: 1px solid #ccc; padding: 8px; border-radius: 6px;">' +
            "<strong>Transcript</strong> (via " + engine + ", confidence " + confidencePct + ")" + uncertainNote + ":<br>" +
            '<em id="voice-transcript-text">' + data.transcript.replace(/</g, "&lt;") + "</em><br>" +
            '<button type="button" id="use-voice-transcript" style="margin-top: 6px;">Use this transcript</button> ' +
            '<span style="font-size: 0.8em; color: #666;">(you can edit it in the Description box afterward)</span>' +
            "</div>";

        document.getElementById("use-voice-transcript").addEventListener("click", function () {
            descriptionEl.value = data.transcript;
            descriptionEl.dispatchEvent(new Event("input"));
        });
    }

    function analyze(blob) {
        resultEl.textContent = "Transcribing...";
        const formData = new FormData();
        formData.append("audio", blob, "recording.webm");
        formData.append("language", getSelectedLanguage());

        fetch("/api/ai/transcribe-audio", { method: "POST", body: formData })
            .then((res) => res.json())
            .then(renderResult)
            .catch(function () {
                resultEl.innerHTML = '<span style="color:#a33;">Could not reach the transcription service - you can still type your complaint below.</span>';
            });
    }

    if (recordBtn && stopBtn && navigator.mediaDevices && window.MediaRecorder) {
        recordBtn.addEventListener("click", function () {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function (stream) {
                    recordedChunks = [];
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = function (e) {
                        if (e.data.size > 0) recordedChunks.push(e.data);
                    };
                    mediaRecorder.onstop = function () {
                        const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
                        stream.getTracks().forEach((t) => t.stop());
                        putFileIntoInput(blob, "recording.webm");
                        showPreview(blob);
                        analyze(blob);
                    };
                    mediaRecorder.start();
                    recordBtn.disabled = true;
                    stopBtn.disabled = false;
                    resultEl.textContent = "Recording...";
                })
                .catch(function () {
                    resultEl.innerHTML = '<span style="color:#a33;">Could not access the microphone. You can upload an audio file instead, or type your complaint directly.</span>';
                });
        });

        stopBtn.addEventListener("click", function () {
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
            }
            recordBtn.disabled = false;
            stopBtn.disabled = true;
        });
    } else if (recordBtn) {
        recordBtn.disabled = true;
        recordBtn.title = "Recording isn't supported in this browser - please upload an audio file instead.";
    }

    audioInput.addEventListener("change", function () {
        const file = audioInput.files[0];
        if (!file) return;
        showPreview(file);
        analyze(file);
    });
})();
