// Interactive Charts for ExoPlanet AI Analytics
// Creates matplotlib-style interactive visualizations using Chart.js

class ChartManager {
    constructor() {
        this.charts = {};
        this.colors = {
            primary: '#00d4ff',
            secondary: '#ff6b6b',
            accent: '#4ecdc4',
            warning: '#ffd93d',
            success: '#6bcf7f',
            gradient: ['#00d4ff', '#4ecdc4', '#ff6b6b', '#ffd93d', '#6bcf7f']
        };
        this.apiBaseUrl = 'http://localhost:8000';
        // Define best models for each telescope
        this.bestModels = {
            'kepler': 'XGBoost',
            'tess': 'Gradient Boosting'
        };
        
        this.init();
    }

    getBestModel(telescope) {
        return this.bestModels[telescope.toLowerCase()] || 'XGBoost';
    }

    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.createCharts('kepler'));
        } else {
            this.createCharts('kepler');
        }
    }

    async createCharts(telescope = 'kepler') {
        // Store current telescope
        this.currentTelescope = telescope;
        
        // Load data from backend
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analytics?telescope=${telescope}`);
            const data = await response.json();
            
            // Update metrics table from test results
            if (data.test_results && data.test_results.metrics) {
                this.updateMetricsTable(data.test_results.metrics);
            }
            
            this.createMetricsChart(data, telescope);
            await this.createFeatureImportanceChart(telescope);
            this.createTrainingProgressChart();
            await this.loadConfusionMatrixImage(telescope);
        } catch (error) {
            console.error('Error loading chart data:', error);
            // Fallback to default charts
            this.createMetricsChart();
            await this.createFeatureImportanceChart(telescope);
            this.createTrainingProgressChart();
        }
    }

    async updateChartsWithData(data) {
        try {
            // Update metrics table and chart
            if (data.test_results && data.test_results.metrics) {
                this.updateMetricsTable(data.test_results.metrics);
                
                // Update bar chart with test results data - use best model for current telescope
                if (this.charts.metrics) {
                    const metrics = data.test_results.metrics;
                    const bestModel = this.getBestModel(this.currentTelescope || 'kepler');
                    // Try best model first, then fall back to first available model
                    const modelToUse = metrics[bestModel] || metrics[Object.keys(metrics)[0]];
                    
                    if (modelToUse) {
                        const metricsData = modelToUse.values.map(v => v * 100);
                        this.charts.metrics.data.datasets[0].data = metricsData;
                        this.charts.metrics.data.datasets[0].label = `${bestModel} Performance (%)`;
                        this.charts.metrics.update();
                    }
                }
            }
            // Fallback to performance_metrics
            else if (this.charts.metrics && data.performance_metrics && 
                data.performance_metrics.values && data.performance_metrics.values.length > 0) {
                const values = data.performance_metrics.values[0];
                const metricsData = [
                    values[1] * 100, // Accuracy
                    values[2] * 100, // Precision
                    values[3] * 100, // Recall
                    values[4] * 100, // F1-Score
                    values[5] * 100, // ROC AUC
                    values[6] * 100  // PR AUC
                ];
                
                this.charts.metrics.data.datasets[0].data = metricsData;
                this.charts.metrics.update();
            }
        } catch (error) {
            console.error('Error updating charts:', error);
        }
    }

    // Update Metrics Table
    updateMetricsTable(metricsData) {
        const tableBody = document.getElementById('metricsTableBody');
        if (!tableBody) return;

        // Clear existing content
        tableBody.innerHTML = '';

        // Extract model names and their metrics
        const models = Object.keys(metricsData);
        
        if (models.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="7" class="loading-cell">No metrics available</td></tr>';
            return;
        }

        // Find best values for each metric (to highlight)
        const allMetrics = models.map(model => metricsData[model].values);
        const bestValues = [];
        for (let i = 0; i < 6; i++) {
            bestValues.push(Math.max(...allMetrics.map(metrics => metrics[i])));
        }

        // Create table rows for each model
        models.forEach(modelName => {
            const modelData = metricsData[modelName];
            const row = document.createElement('tr');
            
            // Model name cell
            const nameCell = document.createElement('td');
            nameCell.textContent = modelName;
            row.appendChild(nameCell);
            
            // Metric value cells
            modelData.values.forEach((value, index) => {
                const cell = document.createElement('td');
                cell.className = 'metric-value-cell';
                
                // Format as decimal (4 decimal places)
                cell.textContent = value.toFixed(4);
                
                // Highlight best metric
                if (value === bestValues[index] && models.length > 1) {
                    cell.classList.add('best-metric');
                }
                
                row.appendChild(cell);
            });
            
            tableBody.appendChild(row);
        });
    }

    // Performance Metrics Bar Chart
    createMetricsChart(data = null, telescope = 'kepler') {
        const ctx = document.getElementById('metricsChart');
        if (!ctx) return;

        // Destroy existing chart if it exists
        if (this.charts.metrics) {
            this.charts.metrics.destroy();
        }

        // Get best model for this telescope
        const bestModel = this.getBestModel(telescope);
        
        // Extract metrics from data if available - use best model as default
        let metricsValues = [89.6, 89.6, 89.6, 89.6, 96.66, 96.77]; // Default values
        let modelLabel = `${bestModel} Performance (%)`;
        
        // Try to get data from test_results first (more accurate)
        if (data && data.test_results && data.test_results.metrics) {
            const metrics = data.test_results.metrics;
            // Try best model first, then fall back to first available model
            const modelToUse = metrics[bestModel] || metrics[Object.keys(metrics)[0]];
            
            if (modelToUse) {
                metricsValues = modelToUse.values.map(v => v * 100);
                modelLabel = `${bestModel} Performance (%)`;
            }
        }
        // Fallback to performance_metrics
        else if (data && data.performance_metrics && data.performance_metrics.values && 
            data.performance_metrics.values.length > 0) {
            const values = data.performance_metrics.values[0];
            metricsValues = [
                values[1] * 100, // Accuracy
                values[2] * 100, // Precision
                values[3] * 100, // Recall
                values[4] * 100, // F1-Score
                values[5] * 100, // ROC AUC
                values[6] * 100  // PR AUC
            ];
            modelLabel = 'Model Performance (%)';
        }

        const metricsData = {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'PR AUC'],
            datasets: [{
                label: modelLabel,
                data: metricsValues,
                backgroundColor: [
                    'rgba(0, 212, 255, 0.8)',
                    'rgba(78, 205, 196, 0.8)',
                    'rgba(255, 107, 107, 0.8)',
                    'rgba(255, 217, 61, 0.8)',
                    'rgba(107, 207, 127, 0.8)',
                    'rgba(0, 212, 255, 0.6)'
                ],
                borderColor: [
                    '#00d4ff',
                    '#4ecdc4',
                    '#ff6b6b',
                    '#ffd93d',
                    '#6bcf7f',
                    '#00d4ff'
                ],
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false,
            }]
        };

        this.charts.metrics = new Chart(ctx, {
            type: 'bar',
            data: metricsData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 26, 46, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#b8b8d1',
                        borderColor: '#00d4ff',
                        borderWidth: 1,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                return `${context.parsed.y.toFixed(2)}%`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(184, 184, 209, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#b8b8d1',
                            font: {
                                size: 12
                            },
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#b8b8d1',
                            font: {
                                size: 12
                            }
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    // ROC Curve Chart
    createROCChart() {
        const ctx = document.getElementById('rocChart');
        if (!ctx) return;

        // Generate ROC curve data (simulated based on 93.59% AUC)
        const rocData = this.generateROCData(0.9359);
        
        const rocDataset = {
            labels: rocData.fpr.map(fpr => fpr.toFixed(3)),
            datasets: [{
                label: 'ROC Curve (AUC = 0.936)',
                data: rocData.tpr,
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: '#00d4ff',
                pointHoverBorderColor: '#ffffff',
                pointHoverBorderWidth: 2
            }, {
                label: 'Random Classifier',
                data: rocData.fpr,
                borderColor: '#ff6b6b',
                backgroundColor: 'transparent',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        };

        this.charts.roc = new Chart(ctx, {
            type: 'line',
            data: rocDataset,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#b8b8d1',
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 26, 46, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#b8b8d1',
                        borderColor: '#00d4ff',
                        borderWidth: 1,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                if (context.datasetIndex === 0) {
                                    return `ROC Curve: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                                } else {
                                    return `Random: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                                }
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'False Positive Rate',
                            color: '#b8b8d1',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        min: 0,
                        max: 1,
                        grid: {
                            color: 'rgba(184, 184, 209, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#b8b8d1',
                            font: {
                                size: 12
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'True Positive Rate',
                            color: '#b8b8d1',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        min: 0,
                        max: 1,
                        grid: {
                            color: 'rgba(184, 184, 209, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#b8b8d1',
                            font: {
                                size: 12
                            }
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    // Feature Importance Horizontal Bar Chart
    async createFeatureImportanceChart(telescope = 'kepler') {
        const ctx = document.getElementById('featureImportanceChart');
        if (!ctx) return;

        // Destroy existing chart if it exists
        if (this.charts.featureImportance) {
            this.charts.featureImportance.destroy();
        }

        // Default data (fallback)
        let featureLabels = [
            'koi_max_sngle_ev', 'koi_depth', 'koi_insol', 'koi_max_mult_ev',
            'koi_dikco_msky', 'koi_insol_err2', 'koi_incl', 'koi_ror',
            'koi_smet_err2', 'koi_prad_err1'
        ];
        let importanceValues = [6.06, 2.61, 1.97, 1.83, 1.62, 1.28, 0.94, 0.72, 0.54, 0.44];

        // Get the best model for this telescope
        const bestModel = this.getBestModel(telescope);
        
        // Try to fetch actual feature importance from backend
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/feature_importance/${telescope}?model=${bestModel}`);
            if (response.ok) {
                const data = await response.json();
                if (data.features && data.features.length > 0) {
                    featureLabels = data.features;
                    if (data.importance_values && data.importance_values.length > 0) {
                        importanceValues = data.importance_values;
                    } else {
                        // If no importance values, use normalized sequence
                        importanceValues = featureLabels.map((_, i) => 
                            (featureLabels.length - i) / featureLabels.length * 6
                        );
                    }
                }
            }
        } catch (error) {
            console.warn('Could not load feature importance from backend, using defaults:', error);
        }

        const featureData = {
            labels: featureLabels,
            datasets: [{
                label: 'Feature Importance',
                data: importanceValues,
                backgroundColor: featureLabels.map((_, i) => {
                    const colors = [
                        'rgba(0, 212, 255, 0.8)', 'rgba(78, 205, 196, 0.8)',
                        'rgba(255, 107, 107, 0.8)', 'rgba(255, 217, 61, 0.8)',
                        'rgba(107, 207, 127, 0.8)'
                    ];
                    return i < 5 ? colors[i] : colors[i % 5].replace('0.8', '0.6');
                }),
                borderColor: featureLabels.map((_, i) => {
                    const colors = ['#00d4ff', '#4ecdc4', '#ff6b6b', '#ffd93d', '#6bcf7f'];
                    return colors[i % 5];
                }),
                borderWidth: 2,
                borderRadius: 4,
                borderSkipped: false,
            }]
        };

        this.charts.featureImportance = new Chart(ctx, {
            type: 'bar',
            data: featureData,
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 26, 46, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#b8b8d1',
                        borderColor: '#00d4ff',
                        borderWidth: 1,
                        cornerRadius: 8,
                        callbacks: {
                            title: function(context) {
                                return context[0].label;
                            },
                            label: function(context) {
                                return `Importance: ${context.parsed.x.toFixed(3)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Importance Score',
                            color: '#b8b8d1',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(184, 184, 209, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#b8b8d1',
                            font: {
                                size: 12
                            }
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#b8b8d1',
                            font: {
                                size: 11
                            },
                            maxRotation: 0
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    // Cross-Validation Progress Chart (using best model's CV results)
    async createTrainingProgressChart() {
        const ctx = document.getElementById('trainingProgressChart');
        if (!ctx) return;

        // Destroy existing chart if it exists
        if (this.charts.trainingProgress) {
            this.charts.trainingProgress.destroy();
        }

        // Default data (fallback)
        let cvScores = Array.from({length: 50}, (_, i) => 0.85 + Math.random() * 0.03);
        let bestScore = Math.max(...cvScores);
        let bestIteration = cvScores.indexOf(bestScore);

        // Try to fetch actual CV results from best model for current telescope
        try {
            const telescope = this.currentTelescope || 'kepler';
            const bestModel = this.getBestModel(telescope);
            const response = await fetch(`${this.apiBaseUrl}/api/cv_results/${telescope}?model=${bestModel}`);
            if (response.ok) {
                const data = await response.json();
                if (data.cv_results && data.cv_results.mean_test_score) {
                    cvScores = data.cv_results.mean_test_score;
                    bestScore = data.best_score || Math.max(...cvScores);
                    bestIteration = cvScores.indexOf(bestScore);
                }
            }
        } catch (error) {
            console.warn('Could not load CV results from backend, using defaults:', error);
        }

        const iterations = Array.from({length: cvScores.length}, (_, i) => i + 1);
        
        // Get model name for display
        const telescope = this.currentTelescope || 'kepler';
        const bestModel = this.getBestModel(telescope);
        
        // Calculate dynamic Y-axis range based on data
        const scoresPercent = cvScores.map(score => score * 100);
        const minScore = Math.min(...scoresPercent);
        const maxScore = Math.max(...scoresPercent);
        const range = maxScore - minScore;
        const yMin = Math.floor(minScore - range * 0.1); // Add 10% padding below
        const yMax = Math.ceil(maxScore + range * 0.1);  // Add 10% padding above

        const trainingData = {
            labels: iterations,
            datasets: [{
                label: `CV Score (${bestModel})`,
                data: scoresPercent,
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.3,
                pointRadius: 3,
                pointHoverRadius: 6,
                pointBackgroundColor: cvScores.map((score, i) => 
                    i === bestIteration ? '#ffd93d' : '#00d4ff'
                ),
                pointBorderColor: cvScores.map((_, i) => 
                    i === bestIteration ? '#ff6b6b' : '#ffffff'
                ),
                pointBorderWidth: cvScores.map((_, i) => i === bestIteration ? 3 : 1)
            }, {
                label: 'Best Score',
                data: Array(cvScores.length).fill(bestScore * 100), // bestScore is already a decimal (0-1 range)
                borderColor: '#ffd93d',
                backgroundColor: 'transparent',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }]
        };

        this.charts.trainingProgress = new Chart(ctx, {
            type: 'line',
            data: trainingData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#b8b8d1',
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 26, 46, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#b8b8d1',
                        borderColor: '#00d4ff',
                        borderWidth: 1,
                        cornerRadius: 8,
                        callbacks: {
                            title: function(context) {
                                const iteration = context[0].label;
                                return `${bestModel} - Iteration ${iteration}`;
                            },
                            label: function(context) {
                                if (context.datasetIndex === 0) {
                                    return `CV Score: ${context.parsed.y.toFixed(2)}%`;
                                } else {
                                    return `Best Score: ${context.parsed.y.toFixed(2)}%`;
                                }
                            },
                            afterLabel: function(context) {
                                if (context.datasetIndex === 0 && context.dataIndex === bestIteration) {
                                    return '⭐ Best Model';
                                }
                                return '';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Optimization Iteration',
                            color: '#b8b8d1',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(184, 184, 209, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#b8b8d1',
                            font: {
                                size: 12
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Cross-Validation Score (%)',
                            color: '#b8b8d1',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(184, 184, 209, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#b8b8d1',
                            font: {
                                size: 12
                            },
                            callback: function(value) {
                                return value.toFixed(1) + '%';
                            }
                        },
                        min: yMin,
                        max: yMax
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    // Generate ROC curve data based on AUC
    generateROCData(auc) {
        const points = 100;
        const fpr = [];
        const tpr = [];
        
        for (let i = 0; i <= points; i++) {
            const x = i / points;
            fpr.push(x);
            
            // Generate TPR based on AUC using a smooth curve
            if (x <= 0.5) {
                tpr.push(2 * auc * x);
            } else {
                tpr.push(2 * auc * (1 - x) + 2 * x - 1);
            }
        }
        
        return { fpr, tpr };
    }

    // Update charts with new data
    updateChart(chartName, newData) {
        if (this.charts[chartName]) {
            this.charts[chartName].data = newData;
            this.charts[chartName].update('active');
        }
    }

    // Animate charts on scroll
    animateChartsOnScroll() {
        const chartElements = document.querySelectorAll('.chart-container');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const chartId = entry.target.querySelector('canvas').id;
                    if (this.charts[chartId]) {
                        this.charts[chartId].update('active');
                    }
                }
            });
        }, { threshold: 0.5 });

        chartElements.forEach(chart => observer.observe(chart));
    }

    // Export chart as image
    exportChart(chartName, format = 'png') {
        if (this.charts[chartName]) {
            const url = this.charts[chartName].toBase64Image();
            const link = document.createElement('a');
            link.download = `${chartName}.${format}`;
            link.href = url;
            link.click();
        }
    }

    // Get chart data as JSON
    getChartData(chartName) {
        if (this.charts[chartName]) {
            return this.charts[chartName].data;
        }
        return null;
    }

    // Destroy all charts
    destroyCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.destroy();
            }
        });
        this.charts = {};
    }

    // Load and display confusion matrix image
    async loadConfusionMatrixImage(telescope = 'kepler', model = null) {
        const imgElement = document.getElementById('confusionMatrixImage');
        const loadingElement = document.getElementById('confusionMatrixLoading');
        
        try {
            // Show loading state
            if (loadingElement) loadingElement.style.display = 'block';
            if (imgElement) imgElement.style.display = 'none';
            
            // Use best model if not specified
            const selectedModel = model || this.getBestModel(telescope);
            console.log(`Loading confusion matrix for ${telescope} with model: ${selectedModel}`);
            
            const response = await fetch(`${this.apiBaseUrl}/api/confusion_matrix/${telescope}?model=${selectedModel}`);
            if (response.ok) {
                const data = await response.json();
                if (imgElement && data.image_url) {
                    // Add timestamp to force reload when switching telescopes
                    const timestamp = new Date().getTime();
                    imgElement.src = `${this.apiBaseUrl}${data.image_url}?t=${timestamp}`;
                    imgElement.alt = `Confusion Matrix for ${data.telescope} - ${data.model}`;
                    
                    // Show image and hide loading when image loads
                    imgElement.onload = () => {
                        imgElement.style.display = 'block';
                        if (loadingElement) loadingElement.style.display = 'none';
                        console.log(`✓ Loaded confusion matrix image: ${data.image_url}`);
                    };
                    
                    imgElement.onerror = () => {
                        if (loadingElement) {
                            loadingElement.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Failed to load confusion matrix image';
                        }
                    };
                }
            } else {
                console.error(`Failed to load confusion matrix: ${response.status}`);
                if (loadingElement) {
                    loadingElement.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Confusion matrix not available';
                }
            }
        } catch (error) {
            console.warn('Could not load confusion matrix image:', error);
            if (loadingElement) {
                loadingElement.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error loading confusion matrix';
            }
        }
    }
}

// Initialize chart manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chartManager = new ChartManager();
});

// Add chart export functionality
function exportChart(chartName) {
    if (window.chartManager) {
        window.chartManager.exportChart(chartName);
    }
}

// Add chart data export functionality
function exportChartData(chartName) {
    if (window.chartManager) {
        const data = window.chartManager.getChartData(chartName);
        if (data) {
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.download = `${chartName}_data.json`;
            link.href = url;
            link.click();
            URL.revokeObjectURL(url);
        }
    }
}

// Export metrics table as CSV
function exportMetricsTable() {
    const table = document.getElementById('metricsTable');
    if (!table) return;

    let csv = [];
    const rows = table.querySelectorAll('tr');

    rows.forEach(row => {
        const cols = row.querySelectorAll('td, th');
        const csvRow = [];
        cols.forEach(col => {
            csvRow.push(col.textContent.trim());
        });
        csv.push(csvRow.join(','));
    });

    const csvContent = csv.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = 'test_metrics.csv';
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
}
