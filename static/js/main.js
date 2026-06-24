/**
 * EstimateNG – Prediction Form Handler
 * Handles form submission, AJAX prediction calls, result rendering,
 * Chart.js feature importance visualisation, and input validation.
 */

'use strict';

(function () {
  const form        = document.getElementById('predictionForm');
  const submitBtn   = document.getElementById('submitBtn');
  const resetBtn    = document.getElementById('resetBtn');
  const newPredBtn  = document.getElementById('newPredictionBtn');
  const resultsCont = document.getElementById('resultsContainer');
  const errorCont   = document.getElementById('errorContainer');
  const errorList   = document.getElementById('errorList');

  let importanceChart = null;

  if (!form) return;  // not on predict page

  // ── Form submission ──────────────────────────────────────
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!clientValidate()) return;

    setLoading(true);
    hideResults();
    hideErrors();

    const payload = buildPayload();

    try {
      const res  = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (data.status === 'error') {
        showErrors(data.errors || ['An unknown error occurred.']);
      } else {
        showResults(data);
      }
    } catch (err) {
      showErrors(['Network error. Please check your connection and try again.']);
    } finally {
      setLoading(false);
    }
  });

  // ── Reset ────────────────────────────────────────────────
  resetBtn.addEventListener('click', () => {
    form.reset();
    form.querySelectorAll('.is-invalid, .is-valid').forEach(el => {
      el.classList.remove('is-invalid', 'is-valid');
    });
    hideResults();
    hideErrors();
    window.scrollTo({ top: form.getBoundingClientRect().top + window.scrollY - 80, behavior: 'smooth' });
  });

  if (newPredBtn) {
    newPredBtn.addEventListener('click', () => {
      hideResults();
      form.querySelectorAll('input[type=number]').forEach(el => el.value = '');
      form.querySelectorAll('select').forEach(el => el.selectedIndex = 0);
      form.querySelectorAll('input[type=checkbox]').forEach(el => el.checked = false);
      window.scrollTo({ top: form.getBoundingClientRect().top + window.scrollY - 80, behavior: 'smooth' });
    });
  }

  // ── Payload builder ──────────────────────────────────────
  function buildPayload() {
    return {
      neighbourhood:  form.neighbourhood.value,
      property_type:  form.property_type.value,
      sqft_living:    parseFloat(form.sqft_living.value),
      bedrooms:       parseInt(form.bedrooms.value, 10),
      bathrooms:      parseInt(form.bathrooms.value, 10),
      parking_spaces: parseInt(form.parking_spaces.value, 10),
      yr_built:       parseInt(form.yr_built.value, 10),
      has_pool:       form.has_pool.checked ? 1 : 0,
      is_gated:       form.is_gated.checked ? 1 : 0,
      has_gym:        form.has_gym.checked  ? 1 : 0,
    };
  }

  // ── Client-side validation (mirrors server rules) ────────
  function clientValidate() {
    let valid = true;

    const rules = [
      { id: 'neighbourhood',  test: v => v !== '',        msg: 'Please select a neighbourhood.' },
      { id: 'property_type',  test: v => v !== '',        msg: 'Please select a property type.' },
      { id: 'sqft_living',    test: v => v >= 10 && v <= 2000, msg: 'Floor area must be 10 – 2,000 m².' },
      { id: 'bedrooms',       test: v => v >= 1 && v <= 20,    msg: 'Bedrooms must be 1 – 20.' },
      { id: 'bathrooms',      test: v => v >= 1 && v <= 15,    msg: 'Bathrooms must be 1 – 15.' },
      { id: 'parking_spaces', test: v => v >= 0 && v <= 10,    msg: 'Parking spaces must be 0 – 10.' },
      { id: 'yr_built',       test: v => v >= 1900 && v <= 2025, msg: 'Year must be 1900 – 2025.' },
    ];

    rules.forEach(({ id, test, msg }) => {
      const el  = document.getElementById(id);
      const val = el.tagName === 'SELECT' ? el.value : parseFloat(el.value);
      const ok  = test(val);
      el.classList.toggle('is-invalid', !ok);
      el.classList.toggle('is-valid',    ok);
      if (!ok) valid = false;
    });

    return valid;
  }

  // ── Results rendering ────────────────────────────────────
  function showResults(data) {
    document.getElementById('predictedPrice').textContent =
      data.predicted_price_formatted || formatNaira(data.predicted_price);

    document.getElementById('confLower').textContent =
      data.confidence_interval?.lower_formatted || formatNaira(data.confidence_interval?.lower);
    document.getElementById('confUpper').textContent =
      data.confidence_interval?.upper_formatted || formatNaira(data.confidence_interval?.upper);

    renderImportanceChart(data.feature_importances || []);

    resultsCont.classList.remove('d-none');
    resultsCont.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function hideResults() {
    resultsCont.classList.add('d-none');
  }

  // ── Feature Importance Chart (Chart.js) ──────────────────
  function renderImportanceChart(importances) {
    const canvas = document.getElementById('importanceChart');
    if (!canvas || importances.length === 0) return;

    const labels = importances.map(d => d.feature);
    const values = importances.map(d => +(d.importance * 100).toFixed(2));

    const colors = values.map((_, i) => {
      const alpha = 1 - (i / values.length) * 0.55;
      return `rgba(31, 56, 100, ${alpha})`;
    });

    if (importanceChart) {
      importanceChart.destroy();
    }

    importanceChart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Relative Importance (%)',
          data: values,
          backgroundColor: colors,
          borderColor: colors.map(c => c.replace(/[\d.]+\)$/, '1)')),
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.parsed.x.toFixed(2)}% influence`,
            },
          },
        },
        scales: {
          x: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Relative Importance (%)',
              font: { size: 12, family: 'Roboto' },
            },
            ticks: { callback: v => `${v}%` },
          },
          y: {
            ticks: { font: { size: 12, family: 'Roboto' } },
          },
        },
      },
    });
  }

  // ── Error display ────────────────────────────────────────
  function showErrors(errors) {
    errorList.innerHTML = '';
    errors.forEach(e => {
      const li = document.createElement('li');
      li.textContent = e;
      errorList.appendChild(li);
    });
    errorCont.classList.remove('d-none');
    errorCont.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function hideErrors() {
    errorCont.classList.add('d-none');
    errorList.innerHTML = '';
  }

  // ── Loading state ────────────────────────────────────────
  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.querySelector('.btn-text').classList.toggle('d-none',  isLoading);
    submitBtn.querySelector('.btn-loading').classList.toggle('d-none', !isLoading);
  }

  // ── Helpers ──────────────────────────────────────────────
  function formatNaira(amount) {
    if (amount == null) return '—';
    return '₦' + Math.round(amount).toLocaleString('en-NG');
  }

})();
