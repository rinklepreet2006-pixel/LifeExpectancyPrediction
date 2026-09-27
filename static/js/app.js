/**
 * VitaPulse WHO Life Expectancy Predictor - Client Application
 * Handles interactive controls, dynamic SVG gauge, real-time predictions,
 * country presets, feature importance charts, and dataset explorer.
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let appMetadata = null;
  let currentCountryProfile = null;
  let debounceTimer = null;
  let explorerCurrentPage = 1;
  let currentPredictionValue = 72.4;

  // DOM Elements
  const presetSelect = document.getElementById('country-preset-select');
  const allCountriesOptgroup = document.getElementById('all-countries-optgroup');
  const btnLoadCountry = document.getElementById('btn-load-country');
  const btnResetMedian = document.getElementById('btn-reset-median');
  const btnRunPrediction = document.getElementById('btn-run-prediction');
  const toggleRealtime = document.getElementById('toggle-realtime');
  const modelSelect = document.getElementById('model-algorithm-select');
  const activeCountryTag = document.getElementById('active-country-tag');

  // Result display elements
  const predictedAgeDisplay = document.getElementById('predicted-age-display');
  const gaugeProgress = document.getElementById('gauge-progress');
  const resultModelName = document.getElementById('result-model-name');
  const resultTierPill = document.getElementById('result-tier-pill');
  const resultTierName = document.getElementById('result-tier-name');
  const resultDeltaPill = document.getElementById('result-delta-pill');
  const resultDeltaText = document.getElementById('result-delta-text');
  const consensusRf = document.getElementById('consensus-rf');
  const consensusGbm = document.getElementById('consensus-gbm');
  const consensusRidge = document.getElementById('consensus-ridge');
  const positiveFactorsList = document.getElementById('positive-factors-list');
  const negativeFactorsList = document.getElementById('negative-factors-list');
  const whoCompareBox = document.getElementById('who-compare-box');
  const actualWhoValue = document.getElementById('actual-who-value');
  const actualWhoDiff = document.getElementById('actual-who-diff');

  // Importance and explorer elements
  const importanceContainer = document.getElementById('importance-bars-container');
  const searchInput = document.getElementById('dataset-search-input');
  const datasetTbody = document.getElementById('dataset-tbody');
  const btnPrevPage = document.getElementById('btn-prev-page');
  const btnNextPage = document.getElementById('btn-next-page');
  const pageCountText = document.getElementById('page-count-text');

  // Feature definitions key list
  const FEATURE_KEYS = [
    'status_developed', 'adult_mortality', 'infant_deaths', 'alcohol',
    'percentage_expenditure', 'hepatitis_b', 'measles', 'bmi',
    'under_five_deaths', 'polio', 'total_expenditure', 'diphtheria',
    'hiv_aids', 'gdp', 'population', 'thinness_1_19_years',
    'thinness_5_9_years', 'income_composition_of_resources', 'schooling'
  ];

  // Initialize application
  async function init() {
    setupTabSwitching();
    setupStatusToggle();
    setupSyncInputs();
    setupEventListeners();
    await fetchMetadata();
    await fetchCountries();
    await fetchDatasetPage(1);
    // Initial prediction with default medians
    runPrediction();
  }

  // 1. Fetch metadata from backend
  async function fetchMetadata() {
    try {
      const res = await fetch('/api/metadata');
      const json = await res.json();
      if (json.status === 'success') {
        appMetadata = json.data;
        renderFeatureImportance(appMetadata.feature_importances);
      }
    } catch (err) {
      console.error('Failed to load metadata:', err);
    }
  }

  // 2. Fetch countries list for preset dropdown
  async function fetchCountries() {
    try {
      const res = await fetch('/api/countries');
      const json = await res.json();
      if (json.status === 'success' && allCountriesOptgroup) {
        allCountriesOptgroup.innerHTML = '';
        json.countries.forEach(country => {
          const opt = document.createElement('option');
          opt.value = country;
          opt.textContent = country;
          allCountriesOptgroup.appendChild(opt);
        });
      }
    } catch (err) {
      console.error('Failed to load countries:', err);
    }
  }

  // 3. Tab switching logic
  function setupTabSwitching() {
    const tabs = document.querySelectorAll('.cat-tab');
    const panels = document.querySelectorAll('.tab-panel');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        panels.forEach(p => p.classList.remove('active'));

        tab.classList.add('active');
        const cat = tab.getAttribute('data-category');
        const targetPanel = document.getElementById(`panel-${cat}`);
        if (targetPanel) {
          targetPanel.classList.add('active');
        }
      });
    });
  }

  // 4. Status Developing / Developed toggle
  function setupStatusToggle() {
    const btnDev = document.getElementById('btn-status-developed');
    const btnDevp = document.getElementById('btn-status-developing');
    const hiddenStatus = document.getElementById('input-status_developed');
    const badgeStatus = document.getElementById('badge-status-val');

    if (btnDev && btnDevp) {
      btnDev.addEventListener('click', () => {
        btnDev.classList.add('active');
        btnDevp.classList.remove('active');
        hiddenStatus.value = '1';
        badgeStatus.textContent = 'Developed';
        badgeStatus.style.color = '#10b981';
        triggerAutoPredict();
      });

      btnDevp.addEventListener('click', () => {
        btnDevp.classList.add('active');
        btnDev.classList.remove('active');
        hiddenStatus.value = '0';
        badgeStatus.textContent = 'Developing';
        badgeStatus.style.color = '#06b6d4';
        triggerAutoPredict();
      });
    }
  }

  // 5. Synchronize range slider and number inputs & update displays
  function setupSyncInputs() {
    FEATURE_KEYS.forEach(key => {
      if (key === 'status_developed') return;

      const slider = document.getElementById(`input-${key}`);
      const numInput = document.getElementById(`num-${key}`);
      const displaySpan = document.getElementById(`val-${key}`);

      if (slider && numInput) {
        // Range slider changed
        slider.addEventListener('input', () => {
          numInput.value = slider.value;
          updateDisplayLabel(key, slider.value);
          triggerAutoPredict();
        });

        // Number input changed
        numInput.addEventListener('input', () => {
          slider.value = numInput.value;
          updateDisplayLabel(key, numInput.value);
          triggerAutoPredict();
        });
      }
    });
  }

  function updateDisplayLabel(key, value) {
    const displaySpan = document.getElementById(`val-${key}`);
    if (!displaySpan) return;

    const v = parseFloat(value);
    switch (key) {
      case 'income_composition_of_resources':
        displaySpan.textContent = v.toFixed(2);
        break;
      case 'schooling':
        displaySpan.textContent = `${v.toFixed(1)} yrs`;
        break;
      case 'gdp':
        displaySpan.textContent = `$${Math.round(v).toLocaleString()}`;
        break;
      case 'percentage_expenditure':
        displaySpan.textContent = `${v.toFixed(1)}%`;
        break;
      case 'total_expenditure':
        displaySpan.textContent = `${v.toFixed(2)}%`;
        break;
      case 'population':
        if (v >= 1e9) displaySpan.textContent = `${(v / 1e9).toFixed(1)}B`;
        else if (v >= 1e6) displaySpan.textContent = `${(v / 1e6).toFixed(1)}M`;
        else if (v >= 1e3) displaySpan.textContent = `${(v / 1e3).toFixed(0)}k`;
        else displaySpan.textContent = `${Math.round(v)}`;
        break;
      case 'hiv_aids':
        displaySpan.textContent = `${v.toFixed(1)} deaths/1k`;
        break;
      case 'adult_mortality':
        displaySpan.textContent = `${Math.round(v)} deaths/1k`;
        break;
      case 'infant_deaths':
      case 'under_five_deaths':
        displaySpan.textContent = `${Math.round(v)} / 1k`;
        break;
      case 'measles':
        displaySpan.textContent = `${Math.round(v).toLocaleString()} cases`;
        break;
      case 'polio':
      case 'diphtheria':
      case 'hepatitis_b':
        displaySpan.textContent = `${Math.round(v)}%`;
        break;
      case 'bmi':
        displaySpan.textContent = `${v.toFixed(1)}`;
        break;
      case 'alcohol':
        displaySpan.textContent = `${v.toFixed(2)} L`;
        break;
      case 'thinness_1_19_years':
      case 'thinness_5_9_years':
        displaySpan.textContent = `${v.toFixed(1)}%`;
        break;
      default:
        displaySpan.textContent = v;
    }
  }

  // 6. Event listeners for buttons and select controls
  function setupEventListeners() {
    if (btnRunPrediction) {
      btnRunPrediction.addEventListener('click', runPrediction);
    }

    if (btnLoadCountry) {
      btnLoadCountry.addEventListener('click', () => {
        const selected = presetSelect.value;
        if (selected) loadCountryProfile(selected);
      });
    }

    if (presetSelect) {
      presetSelect.addEventListener('change', () => {
        if (presetSelect.value) {
          loadCountryProfile(presetSelect.value);
        }
      });
    }

    if (btnResetMedian) {
      btnResetMedian.addEventListener('click', resetToGlobalMedian);
    }

    if (modelSelect) {
      modelSelect.addEventListener('change', () => {
        runPrediction();
      });
    }

    // Dataset search
    if (searchInput) {
      searchInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          explorerCurrentPage = 1;
          fetchDatasetPage(1, searchInput.value);
        }, 300);
      });
    }

    if (btnPrevPage) {
      btnPrevPage.addEventListener('click', () => {
        if (explorerCurrentPage > 1) {
          explorerCurrentPage--;
          fetchDatasetPage(explorerCurrentPage, searchInput ? searchInput.value : '');
        }
      });
    }

    if (btnNextPage) {
      btnNextPage.addEventListener('click', () => {
        explorerCurrentPage++;
        fetchDatasetPage(explorerCurrentPage, searchInput ? searchInput.value : '');
      });
    }
  }

  function triggerAutoPredict() {
    if (toggleRealtime && toggleRealtime.checked) {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        runPrediction();
      }, 120);
    }
  }

  // 7. Load Country Profile
  async function loadCountryProfile(countryName) {
    try {
      const res = await fetch(`/api/country/${encodeURIComponent(countryName)}`);
      const json = await res.json();
      if (json.status !== 'success') return;

      const profile = json.data;
      currentCountryProfile = profile;

      // Update status button
      const isDeveloped = profile.status_developed === 1;
      const btnDev = document.getElementById('btn-status-developed');
      const btnDevp = document.getElementById('btn-status-developing');
      const hiddenStatus = document.getElementById('input-status_developed');
      const badgeStatus = document.getElementById('badge-status-val');

      if (isDeveloped) {
        btnDev.classList.add('active');
        btnDevp.classList.remove('active');
        hiddenStatus.value = '1';
        badgeStatus.textContent = 'Developed';
        badgeStatus.style.color = '#10b981';
      } else {
        btnDevp.classList.add('active');
        btnDev.classList.remove('active');
        hiddenStatus.value = '0';
        badgeStatus.textContent = 'Developing';
        badgeStatus.style.color = '#06b6d4';
      }

      // Populate all indicators
      for (const [key, val] of Object.entries(profile.features)) {
        if (key === 'status_developed') continue;

        const slider = document.getElementById(`input-${key}`);
        const numInput = document.getElementById(`num-${key}`);
        if (slider) slider.value = val;
        if (numInput) numInput.value = val;
        updateDisplayLabel(key, val);
      }

      // Update active country tag
      if (activeCountryTag) {
        activeCountryTag.textContent = `Active Profile: ${profile.country} (Latest WHO: ${profile.year})`;
      }

      // Show WHO comparison card
      if (whoCompareBox) {
        whoCompareBox.style.display = 'block';
        actualWhoValue.textContent = `${profile.actual_life_expectancy} years`;
      }

      // Run prediction
      runPrediction();

    } catch (err) {
      console.error(`Error loading profile for ${countryName}:`, err);
    }
  }

  // 8. Reset all fields to global median values
  function resetToGlobalMedian() {
    if (!appMetadata) return;

    currentCountryProfile = null;
    if (presetSelect) presetSelect.value = '';
    if (activeCountryTag) activeCountryTag.textContent = 'Using: Global Baseline Profile';
    if (whoCompareBox) whoCompareBox.style.display = 'none';

    // Reset status to Developing (majority in dataset)
    const btnDev = document.getElementById('btn-status-developed');
    const btnDevp = document.getElementById('btn-status-developing');
    const hiddenStatus = document.getElementById('input-status_developed');
    const badgeStatus = document.getElementById('badge-status-val');
    btnDevp.classList.add('active');
    btnDev.classList.remove('active');
    hiddenStatus.value = '0';
    badgeStatus.textContent = 'Developing';

    // Reset numeric inputs
    FEATURE_KEYS.forEach(key => {
      if (key === 'status_developed') return;
      const stats = appMetadata.feature_stats[key];
      if (stats) {
        const med = stats.median;
        const slider = document.getElementById(`input-${key}`);
        const numInput = document.getElementById(`num-${key}`);
        if (slider) slider.value = med;
        if (numInput) numInput.value = med;
        updateDisplayLabel(key, med);
      }
    });

    runPrediction();
  }

  // 9. Collect inputs & invoke prediction API
  async function runPrediction() {
    const payloadFeatures = {};

    FEATURE_KEYS.forEach(key => {
      if (key === 'status_developed') {
        const hiddenStatus = document.getElementById('input-status_developed');
        payloadFeatures[key] = hiddenStatus ? parseInt(hiddenStatus.value, 10) : 0;
      } else {
        const input = document.getElementById(`input-${key}`);
        payloadFeatures[key] = input ? parseFloat(input.value) : 0;
      }
    });

    const selectedModel = modelSelect ? modelSelect.value : 'Random Forest';

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          features: payloadFeatures
        })
      });

      const json = await res.json();
      if (json.status === 'success') {
        updatePredictionUI(json);
      }
    } catch (err) {
      console.error('Prediction request failed:', err);
    }
  }

  // 10. Update UI with prediction results
  function updatePredictionUI(result) {
    const newAge = result.prediction;
    animateAgeCount(currentPredictionValue, newAge, 400);
    currentPredictionValue = newAge;

    // Update SVG gauge arc
    // Total circumference arc length = ~235.6 (semi-circle of radius 75)
    // Range maps from 35 years (empty) to 90 years (full)
    const minAge = 35.0;
    const maxAge = 90.0;
    const clamped = Math.max(minAge, Math.min(maxAge, newAge));
    const ratio = (clamped - minAge) / (maxAge - minAge);
    const maxDash = 235.6;
    const offset = maxDash * (1 - ratio);
    
    if (gaugeProgress) {
      gaugeProgress.style.strokeDashoffset = offset;
      // Change color based on age
      if (newAge >= 80) {
        gaugeProgress.style.stroke = '#10b981'; // Emerald
      } else if (newAge >= 74) {
        gaugeProgress.style.stroke = '#06b6d4'; // Cyan
      } else if (newAge >= 66) {
        gaugeProgress.style.stroke = '#f59e0b'; // Amber
      } else {
        gaugeProgress.style.stroke = '#f43f5e'; // Rose
      }
    }

    // Model name tag
    if (resultModelName) {
      resultModelName.textContent = result.model_used;
    }

    // Tier badge
    if (resultTierPill && resultTierName) {
      resultTierName.textContent = `${result.tier.name} (${result.tier.rank})`;
      resultTierPill.style.color = result.tier.color;
      resultTierPill.style.borderColor = result.tier.color;
      resultTierPill.style.background = `${result.tier.color}22`;
    }

    // Delta comparison pill
    if (resultDeltaPill && resultDeltaText) {
      const delta = result.benchmark_comparison.delta;
      const sign = delta >= 0 ? '+' : '';
      resultDeltaText.textContent = `${sign}${delta} yrs vs WHO Global Mean (69.2y)`;
      if (delta >= 0) {
        resultDeltaText.style.color = '#34d399';
      } else {
        resultDeltaText.style.color = '#f87171';
      }
    }

    // Cross-model consensus
    if (consensusRf) consensusRf.textContent = `${result.model_comparisons['Random Forest']}y`;
    if (consensusGbm) consensusGbm.textContent = `${result.model_comparisons['Gradient Boosting']}y`;
    if (consensusRidge) consensusRidge.textContent = `${result.model_comparisons['Ridge']}y`;

    // Explainable AI drivers (Positive & Negative)
    renderDrivers(result.top_drivers);

    // If country preset is active, update actual WHO difference
    if (currentCountryProfile && whoCompareBox && actualWhoDiff) {
      const actualVal = currentCountryProfile.actual_life_expectancy;
      const diff = Math.abs(newAge - actualVal).toFixed(1);
      const acc = Math.max(0, 100 - (diff / actualVal) * 100).toFixed(1);
      actualWhoDiff.textContent = `Model absolute error: ${diff} yrs (${acc}% congruence)`;
    }
  }

  // Smooth number counter animation
  function animateAgeCount(start, end, duration) {
    const startTime = performance.now();
    function step(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3); // Ease out cubic
      const current = start + (end - start) * ease;
      if (predictedAgeDisplay) {
        predictedAgeDisplay.textContent = current.toFixed(1);
      }
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        if (predictedAgeDisplay) {
          predictedAgeDisplay.textContent = end.toFixed(1);
        }
      }
    }
    requestAnimationFrame(step);
  }

  // Render explainability chips
  function renderDrivers(drivers) {
    if (positiveFactorsList) {
      positiveFactorsList.innerHTML = '';
      if (drivers.positive && drivers.positive.length > 0) {
        drivers.positive.forEach(d => {
          const chip = document.createElement('div');
          chip.className = 'factor-chip positive';
          chip.innerHTML = `
            <span class="chip-icon">✔</span>
            <span class="chip-name">${d.label}: ${formatDriverVal(d)}</span>
          `;
          positiveFactorsList.appendChild(chip);
        });
      } else {
        positiveFactorsList.innerHTML = '<span class="text-muted" style="font-size: 0.75rem;">None prominent</span>';
      }
    }

    if (negativeFactorsList) {
      negativeFactorsList.innerHTML = '';
      if (drivers.negative && drivers.negative.length > 0) {
        drivers.negative.forEach(d => {
          const chip = document.createElement('div');
          chip.className = 'factor-chip negative';
          chip.innerHTML = `
            <span class="chip-icon">⚠</span>
            <span class="chip-name">${d.label}: ${formatDriverVal(d)}</span>
          `;
          negativeFactorsList.appendChild(chip);
        });
      } else {
        negativeFactorsList.innerHTML = '<span class="text-muted" style="font-size: 0.75rem;">None prominent</span>';
      }
    }
  }

  function formatDriverVal(d) {
    const val = d.user_val;
    if (d.feature === 'gdp') return `$${Math.round(val).toLocaleString()}`;
    if (d.feature.includes('expenditure') || d.feature.includes('polio') || d.feature.includes('diphtheria')) return `${val}%`;
    return `${val}`;
  }

  // 11. Render Feature Importance visual bars
  function renderFeatureImportance(importances) {
    if (!importanceContainer) return;
    importanceContainer.innerHTML = '';

    // Show top 8 features for clarity
    const topFeatures = importances.slice(0, 8);

    topFeatures.forEach(item => {
      const isNegative = ['hiv_aids', 'adult_mortality', 'under_five_deaths'].includes(item.feature);
      const row = document.createElement('div');
      row.className = 'imp-bar-row';
      row.innerHTML = `
        <div class="imp-bar-meta">
          <span class="imp-bar-label">
            ${item.label}
            <span class="imp-bar-category">(${item.category})</span>
          </span>
          <span class="imp-bar-percent">${item.percentage}%</span>
        </div>
        <div class="imp-bar-track">
          <div class="imp-bar-fill ${isNegative ? 'fill-danger' : ''}" style="width: 0%;"></div>
        </div>
      `;
      importanceContainer.appendChild(row);

      // Animate width after render
      setTimeout(() => {
        const fill = row.querySelector('.imp-bar-fill');
        if (fill) fill.style.width = `${Math.min(100, item.percentage * 1.5)}%`;
      }, 80);
    });
  }

  // 12. Dataset Explorer fetch and table rendering
  async function fetchDatasetPage(page = 1, searchQuery = '') {
    try {
      const url = `/api/dataset/sample?page=${page}&per_page=8&search=${encodeURIComponent(searchQuery)}`;
      const res = await fetch(url);
      const json = await res.json();
      if (json.status !== 'success') return;

      renderDatasetTable(json.records);

      if (pageCountText) {
        const start = (json.page - 1) * json.per_page + 1;
        const end = Math.min(json.page * json.per_page, json.total);
        pageCountText.textContent = `Showing ${start} - ${end} of ${json.total.toLocaleString()} records`;
      }

      if (btnPrevPage) btnPrevPage.disabled = json.page <= 1;
      if (btnNextPage) btnNextPage.disabled = json.page >= json.total_pages;

    } catch (err) {
      console.error('Failed to fetch dataset page:', err);
    }
  }

  function renderDatasetTable(records) {
    if (!datasetTbody) return;
    datasetTbody.innerHTML = '';

    if (records.length === 0) {
      datasetTbody.innerHTML = `
        <tr>
          <td colspan="10" style="text-align: center; color: var(--text-muted); padding: 2rem;">
            No matching records found.
          </td>
        </tr>
      `;
      return;
    }

    records.forEach(row => {
      const tr = document.createElement('tr');
      const isDeveloped = row.status === 'Developed';

      tr.innerHTML = `
        <td><strong>${row.country}</strong></td>
        <td>${row.year}</td>
        <td>
          <span style="font-size: 0.74rem; font-weight: 600; color: ${isDeveloped ? 'var(--accent-emerald)' : 'var(--accent-cyan)'}">
            ${row.status}
          </span>
        </td>
        <td><strong style="color: #fff;">${row.life_expectancy} yrs</strong></td>
        <td>${row.adult_mortality}</td>
        <td>${row.infant_deaths}</td>
        <td>${row.hiv_aids}</td>
        <td>${row.bmi}</td>
        <td>${row.schooling} yrs</td>
        <td>
          <button class="btn-table-action" data-country="${row.country}" title="Load this nation's indicators into the predictor">
            ⚡ Inspect
          </button>
        </td>
      `;

      // Attach inspect action
      const btn = tr.querySelector('.btn-table-action');
      btn.addEventListener('click', () => {
        const cName = btn.getAttribute('data-country');
        if (presetSelect) presetSelect.value = cName;
        loadCountryProfile(cName);
        // Smooth scroll up to predictor
        const predSec = document.getElementById('predictor-section');
        if (predSec) predSec.scrollIntoView({ behavior: 'smooth' });
      });

      datasetTbody.appendChild(tr);
    });
  }

  // Start initialization
  init();
});
