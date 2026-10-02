// Shows a rough resolution-time estimate once category + priority are
// both set. Purely informational (no "Use this" button - there's nothing
// to apply to the form). Explicitly worded as an estimate, never a
// promised deadline, per the requirement that these numbers must not be
// presented as guarantees.

(function () {
    const categoryEl = document.getElementById("category");
    const priorityEl = document.getElementById("priority");
    const estimateEl = document.getElementById("ai-resolution-estimate");

    if (!categoryEl || !priorityEl || !estimateEl) return;

    function maybeShowEstimate() {
        const category = categoryEl.value;
        const priority = priorityEl.value;

        if (!category || !priority) {
            estimateEl.textContent = "";
            return;
        }

        fetch("/api/ai/predict-resolution-time", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ category: category, priority: priority }),
        })
            .then((res) => res.json())
            .then((data) => {
                if (!data.available) {
                    estimateEl.textContent = "";
                    return;
                }
                let note = data.low_confidence
                    ? " (based on a bootstrap model — treat as illustrative)"
                    : "";
                estimateEl.textContent =
                    "Estimated resolution time: ~" + data.predicted_days +
                    " days" + note + ". This is an estimate, not a guaranteed deadline.";
            })
            .catch(function () {
                estimateEl.textContent = "";
            });
    }

    categoryEl.addEventListener("change", maybeShowEstimate);
    priorityEl.addEventListener("change", maybeShowEstimate);
})();
