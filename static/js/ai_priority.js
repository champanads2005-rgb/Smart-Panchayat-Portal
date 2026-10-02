// Calls /api/ai/predict-priority once both a description and a category
// are present, and shows a suggested priority with real model confidence
// and the factors that drove it. The citizen must click "Use this" to
// apply it. Recurrence count is computed server-side from real complaint
// history for the citizen's village - never trust a client-supplied value
// for that.

(function () {
    const descriptionEl = document.getElementById("description");
    const categoryEl = document.getElementById("category");
    const priorityEl = document.getElementById("priority");
    const suggestionEl = document.getElementById("ai-priority-suggestion");

    if (!descriptionEl || !categoryEl || !priorityEl || !suggestionEl) return;

    let debounceTimer = null;

    function maybeSuggestPriority() {
        clearTimeout(debounceTimer);
        const text = descriptionEl.value.trim();
        const category = categoryEl.value;

        if (text.length < 12 || !category) {
            suggestionEl.textContent = "";
            return;
        }

        debounceTimer = setTimeout(function () {
            fetch("/api/ai/predict-priority", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text, category: category }),
            })
                .then((res) => res.json())
                .then((data) => {
                    if (!data.available) {
                        suggestionEl.textContent = "";
                        return;
                    }

                    const confidencePct = data.confidence !== null
                        ? Math.round(data.confidence * 100) + "%"
                        : "n/a";

                    let factorsText = "";
                    if (data.top_factors && data.top_factors.length > 0) {
                        factorsText = " — driven by: " + data.top_factors.join(", ");
                    }

                    suggestionEl.innerHTML =
                        "AI suggests priority: <strong>" + data.priority + "</strong> " +
                        "(confidence " + confidencePct + ")" + factorsText + " " +
                        '<button type="button" id="use-ai-priority">Use this</button>';

                    document.getElementById("use-ai-priority").addEventListener("click", function () {
                        priorityEl.value = data.priority;
                    });
                })
                .catch(function () {
                    suggestionEl.textContent = "";
                });
        }, 700);
    }

    descriptionEl.addEventListener("input", maybeSuggestPriority);
    categoryEl.addEventListener("change", maybeSuggestPriority);
})();
