// Calls /api/ai/classify with the complaint description and shows a
// suggested category with its real model confidence. The citizen must
// click "Use this" to apply it - nothing is auto-submitted or hidden.
// If the AI service is unavailable, the form still works with manual
// category selection (graceful fallback, no fake result is shown).

(function () {
    const descriptionEl = document.getElementById("description");
    const categoryEl = document.getElementById("category");
    const suggestionEl = document.getElementById("ai-suggestion");

    if (!descriptionEl || !categoryEl || !suggestionEl) return;

    let debounceTimer = null;

    descriptionEl.addEventListener("input", function () {
        clearTimeout(debounceTimer);
        const text = descriptionEl.value.trim();

        if (text.length < 12) {
            suggestionEl.textContent = "";
            return;
        }

        debounceTimer = setTimeout(function () {
            fetch("/api/ai/classify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text }),
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

                    suggestionEl.innerHTML =
                        "AI suggests category: <strong>" + data.category + "</strong> " +
                        "(confidence " + confidencePct + ") " +
                        '<button type="button" id="use-ai-suggestion">Use this</button>';

                    document.getElementById("use-ai-suggestion").addEventListener("click", function () {
                        categoryEl.value = data.category;
                    });
                })
                .catch(function () {
                    // Fail silently - manual category selection still works.
                    suggestionEl.textContent = "";
                });
        }, 600);
    });
})();
