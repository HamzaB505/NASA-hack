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
        
        this.init();
    }

    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.createCharts());
        } else {
            this.createCharts();
        }
    }

    async createCharts(telescope = 'kepler') {
        // Load data from backend
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analytics?telescope=${telescope}`);
            const data = await response.json();
            
            // Update metrics table from test results
            if (data.test_results && data.test_results.metrics) {
                this.updateMetricsTable(data.test_results.metrics);
            }
            
            this.createMetricsChart(data);
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
                
                // Update bar chart with test results data
                if (this.charts.metrics) {
                    const models = Object.keys(data.test_results.metrics);
                    if (models.length > 0) {
                        const firstModel = data.test_results.metrics[models[0]];
                        const metricsData = firstModel.values.map(v => v * 100);
                        this.charts.metrics.data.datasets[0].data = metricsData;
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
    createMetricsChart(data = null) {
        const ctx = document.getElementById('metricsChart');
        if (!ctx) return;

        // Extract metrics from data if available
        let metricsValues = [86.25, 86.50, 86.25, 86.24, 93.59, 93.25];
        
        // Try to get data from test_results first (more accurate)
        if (data && data.test_results && data.test_results.metrics) {
            const models = Object.keys(data.test_results.metrics);
            if (models.length > 0) {
                const firstModel = data.test_results.metrics[models[0]];
                metricsValues = firstModel.values.map(v => v * 100);
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
        }

        const metricsData = {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'PR AUC'],
            datasets: [{
                label: 'Model Performance (%)',
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

        // Default data (fallback)
        let featureLabels = [
            'koi_max_sngle_ev', 'koi_depth', 'koi_insol', 'koi_max_mult_ev',
            'koi_dikco_msky', 'koi_insol_err2', 'koi_incl', 'koi_ror',
            'koi_smet_err2', 'koi_prad_err1'
        ];
        let importanceValues = [6.06, 2.61, 1.97, 1.83, 1.62, 1.28, 0.94, 0.72, 0.54, 0.44];

        // Try to fetch actual feature importance from backend
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/feature_importance/${telescope}`);
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

    // Training Progress Line Chart
    createTrainingProgressChart() {
        const ctx = document.getElementById('trainingProgressChart');
        if (!ctx) return;

        // Simulate training progress data
        const epochs = Array.from({length: 20}, (_, i) => i + 1);
        const trainLoss = epochs.map(epoch => 0.8 * Math.exp(-epoch/5) + 0.1 + Math.random() * 0.05);
        const valLoss = epochs.map(epoch => 0.7 * Math.exp(-epoch/6) + 0.15 + Math.random() * 0.05);
        const trainAcc = epochs.map(epoch => 0.5 + 0.4 * (1 - Math.exp(-epoch/4)) + Math.random() * 0.02);
        const valAcc = epochs.map(epoch => 0.5 + 0.35 * (1 - Math.exp(-epoch/5)) + Math.random() * 0.02);

        const trainingData = {
            labels: epochs,
            datasets: [{
                label: 'Training Loss',
                data: trainLoss,
                borderColor: '#ff6b6b',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1,
                yAxisID: 'y'
            }, {
                label: 'Validation Loss',
                data: valLoss,
                borderColor: '#ffd93d',
                backgroundColor: 'rgba(255, 217, 61, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1,
                yAxisID: 'y'
            }, {
                label: 'Training Accuracy',
                data: trainAcc,
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1,
                yAxisID: 'y1'
            }, {
                label: 'Validation Accuracy',
                data: valAcc,
                borderColor: '#4ecdc4',
                backgroundColor: 'rgba(78, 205, 196, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1,
                yAxisID: 'y1'
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
                            label: function(context) {
                                if (context.datasetIndex < 2) {
                                    return `${context.dataset.label}: ${context.parsed.y.toFixed(4)}`;
                                } else {
                                    return `${context.dataset.label}: ${(context.parsed.y * 100).toFixed(2)}%`;
                                }
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch',
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
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Loss',
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
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Accuracy',
                            color: '#b8b8d1',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                        ticks: {
                            color: '#b8b8d1',
                            font: {
                                size: 12
                            },
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
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
    async loadConfusionMatrixImage(telescope = 'kepler') {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/confusion_matrix/${telescope}`);
            if (response.ok) {
                const data = await response.json();
                const imgElement = document.getElementById('confusionMatrixImage');
                if (imgElement && data.image_url) {
                    imgElement.src = `${this.apiBaseUrl}${data.image_url}`;
                    imgElement.alt = `Confusion Matrix for ${data.telescope}`;
                    imgElement.style.display = 'block';
                }
            }
        } catch (error) {
            console.warn('Could not load confusion matrix image:', error);
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
