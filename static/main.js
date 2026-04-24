document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const textInput = document.getElementById('text-input');
    const resultsSection = document.getElementById('results-section');
    const btnLoader = document.getElementById('btn-loader');
    
    const elements = {
        badge: document.getElementById('sentiment-badge'),
        valNeg: document.getElementById('val-neg'),
        valNeu: document.getElementById('val-neu'),
        valPos: document.getElementById('val-pos'),
        barNeg: document.getElementById('bar-neg'),
        barNeu: document.getElementById('bar-neu'),
        barPos: document.getElementById('bar-pos')
    };

    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) {
            textInput.style.borderColor = 'var(--neg-color)';
            textInput.style.boxShadow = '0 0 0 3px rgba(239, 68, 68, 0.2)';
            setTimeout(() => {
                textInput.style.borderColor = '';
                textInput.style.boxShadow = '';
            }, 1000);
            return;
        }

        // Show loading state
        const btnText = analyzeBtn.querySelector('span');
        btnText.style.display = 'none';
        btnLoader.style.display = 'block';
        analyzeBtn.disabled = true;
        
        // Hide previous results
        resultsSection.classList.add('hidden');
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) throw new Error('Prediction failed API side');
            
            const data = await response.json();
            updateUI(data);

        } catch (error) {
            console.error(error);
            alert("Error running the analysis module.");
        } finally {
            // Revert button state
            btnText.style.display = 'block';
            btnLoader.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });

    clearBtn.addEventListener('click', () => {
        textInput.value = '';
        resultsSection.classList.add('hidden');
        resetBars();
    });

    function updateUI(data) {
        // Unhide section
        resultsSection.classList.remove('hidden');

        // Color coding map
        const colorMap = {
            'Negative': 'var(--neg-color)',
            'Neutral': 'var(--neu-color)',
            'Positive': 'var(--pos-color)'
        };

        const bgMap = {
            'Negative': 'rgba(239, 68, 68, 0.15)',
            'Neutral': 'rgba(234, 179, 8, 0.15)',
            'Positive': 'rgba(16, 185, 129, 0.15)'
        };

        // Update badge
        elements.badge.textContent = `Dominant: ${data.sentiment} (${data.confidence}%)`;
        elements.badge.style.color = colorMap[data.sentiment];
        elements.badge.style.background = bgMap[data.sentiment];

        // Animate numbers
        animateValue(elements.valNeg, 0, data.probabilities.Negative, 1000);
        animateValue(elements.valNeu, 0, data.probabilities.Neutral, 1000);
        animateValue(elements.valPos, 0, data.probabilities.Positive, 1000);

        // Update bars
        setTimeout(() => {
            elements.barNeg.style.width = `${data.probabilities.Negative}%`;
            elements.barNeu.style.width = `${data.probabilities.Neutral}%`;
            elements.barPos.style.width = `${data.probabilities.Positive}%`;
        }, 100);
    }

    function resetBars() {
        elements.barNeg.style.width = '0%';
        elements.barNeu.style.width = '0%';
        elements.barPos.style.width = '0%';
        elements.valNeg.textContent = '0%';
        elements.valNeu.textContent = '0%';
        elements.valPos.textContent = '0%';
    }

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = (progress * (end - start) + start).toFixed(1) + '%';
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
