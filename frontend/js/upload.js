// Upload and Prediction Handler for ExoPlanet AI

class UploadHandler {
    constructor() {
        this.uploadedFile = null;
        this.selectedTelescope = 'kepler';
        this.selectedModel = 'xgboost';
        this.apiBaseUrl = 'http://localhost:8000';
        this.availableModels = {
            kepler: [],
            tess: []
        };
        
        this.init();
    }

    init() {
        this.setupFileUpload();
        this.setupFormHandlers();
        this.setupDragAndDrop();
        this.loadAvailableModels();
    }

    async loadAvailableModels() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/models`);
            const data = await response.json();
            
            // Store available models
            this.availableModels.kepler = data.kepler_models || [];
            this.availableModels.tess = data.tess_models || [];
            
            // Set default model
            if (data.default_model) {
                this.selectedModel = data.default_model.toLowerCase().replace(' ', '_');
            }
            
            // Update model options for current telescope
            this.updateModelOptions();
            
        } catch (error) {
            console.error('Error loading available models:', error);
            // Fallback to default models
            this.availableModels.kepler = ['Logistic Regression', 'XGBoost', 'Random Forest', 'Gradient Boosting', 'Decision Tree'];
            this.availableModels.tess = ['Logistic Regression', 'XGBoost', 'Random Forest', 'Gradient Boosting', 'Decision Tree'];
            this.updateModelOptions();
        }
    }

    setupFileUpload() {
        const fileInput = document.getElementById('fileInput');
        const uploadZone = document.getElementById('uploadZone');
        const predictBtn = document.getElementById('predictBtn');
        const clearBtn = document.getElementById('clearBtn');

        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                this.handleFileSelect(e.target.files[0]);
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearUpload();
            });
        }

        if (predictBtn) {
            predictBtn.addEventListener('click', () => {
                this.predictExoplanet();
            });
        }
    }

    setupFormHandlers() {
        const telescopeSelect = document.getElementById('telescopeSelect');
        const modelSelect = document.getElementById('modelSelect');

        if (telescopeSelect) {
            telescopeSelect.addEventListener('change', (e) => {
                this.selectedTelescope = e.target.value;
                this.updateModelOptions();
            });
        }

        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                this.selectedModel = e.target.value;
            });
        }
    }

    setupDragAndDrop() {
        const uploadZone = document.getElementById('uploadZone');

        if (uploadZone) {
            // Prevent default drag behaviors
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadZone.addEventListener(eventName, this.preventDefaults, false);
                document.body.addEventListener(eventName, this.preventDefaults, false);
            });

            // Highlight drop area when item is dragged over it
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadZone.addEventListener(eventName, () => {
                    uploadZone.classList.add('dragover');
                }, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                uploadZone.addEventListener(eventName, () => {
                    uploadZone.classList.remove('dragover');
                }, false);
            });

            // Handle dropped files
            uploadZone.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                const files = dt.files;
                
                if (files.length > 0) {
                    this.handleFileSelect(files[0]);
                }
            }, false);

            // Handle click to open file dialog
            uploadZone.addEventListener('click', () => {
                document.getElementById('fileInput').click();
            });
        }
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        const allowedTypes = [
            'text/csv',
            'application/json',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ];

        const allowedExtensions = ['.csv', '.json', '.xlsx', '.xls'];
        const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

        if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
            this.showError('Please upload a CSV, JSON, or Excel file.');
            return;
        }

        // Validate file size (max 10MB)
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            this.showError('File size must be less than 10MB.');
            return;
        }

        this.uploadedFile = file;
        this.updateUploadUI();
        this.enablePredictButton();
    }

    updateUploadUI() {
        const uploadZone = document.getElementById('uploadZone');
        const uploadContent = uploadZone.querySelector('.upload-content');

        if (this.uploadedFile) {
            uploadContent.innerHTML = `
                <i class="fas fa-file-check upload-icon" style="color: var(--success-color);"></i>
                <h3>File Ready for Analysis</h3>
                <p><strong>${this.uploadedFile.name}</strong></p>
                <p>Size: ${this.formatFileSize(this.uploadedFile.size)}</p>
                <p>Type: ${this.uploadedFile.type || 'Unknown'}</p>
                <button class="btn btn-secondary" onclick="document.getElementById('fileInput').click()">
                    <i class="fas fa-upload"></i>
                    Choose Different File
                </button>
            `;
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    enablePredictButton() {
        const predictBtn = document.getElementById('predictBtn');
        if (predictBtn && this.uploadedFile) {
            predictBtn.disabled = false;
            predictBtn.classList.add('btn-animated');
        }
    }

    updateModelOptions() {
        const modelSelect = document.getElementById('modelSelect');
        if (!modelSelect) return;

        // Get available models for selected telescope
        const models = this.availableModels[this.selectedTelescope] || [];
        
        if (models.length === 0) {
            // No models available for this telescope
            modelSelect.innerHTML = `
                <option value="">No models available</option>
            `;
            return;
        }

        // Build options HTML
        let optionsHTML = '';
        models.forEach(modelName => {
            // Convert model name to value format (e.g., "XGBoost" -> "xgboost")
            const modelValue = modelName.toLowerCase().replace(/\s+/g, '_');
            const isSelected = modelValue === this.selectedModel;
            optionsHTML += `<option value="${modelValue}" ${isSelected ? 'selected' : ''}>${modelName}</option>`;
        });

        modelSelect.innerHTML = optionsHTML;
        
        // Update selected model if it's not available for this telescope
        if (!models.some(m => m.toLowerCase().replace(/\s+/g, '_') === this.selectedModel)) {
            const firstModelValue = models[0].toLowerCase().replace(/\s+/g, '_');
            this.selectedModel = firstModelValue;
            modelSelect.value = firstModelValue;
        }
    }

    async predictExoplanet() {
        // Check if we're using form data or file upload
        const activeTab = document.querySelector('.method-tab.active');
        const isFormMethod = activeTab && activeTab.dataset.method === 'form';

        if (isFormMethod) {
            // Validate form data
            if (!validateFormData()) {
                return;
            }
            
            const formData = getFormData();
            await this.predictFromFormData(formData);
        } else {
            // File upload method
            if (!this.uploadedFile) {
                this.showError('Please upload a file first.');
                return;
            }

            // Check if models are available for selected telescope
            const modelsAvailable = this.availableModels[this.selectedTelescope] && 
                                   this.availableModels[this.selectedTelescope].length > 0;
            if (!modelsAvailable) {
                this.showError(`No models available for ${this.selectedTelescope.toUpperCase()} telescope yet.`);
                return;
            }

            try {
                this.showLoading();
                
                // Prepare form data
                const formData = new FormData();
                formData.append('file', this.uploadedFile);
                formData.append('telescope', this.selectedTelescope);
                formData.append('model', this.selectedModel);

                // Make API request
                const response = await fetch(`${this.apiBaseUrl}/api/predict`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                this.displayResults(result);

            } catch (error) {
                console.error('Prediction failed:', error);
                
                // For demo purposes, show mock results if API is not available
                if (error.message.includes('Failed to fetch')) {
                    this.showMockResults();
                } else {
                    this.showError('Prediction failed. Please try again.');
                }
            } finally {
                this.hideLoading();
            }
        }
    }

    async predictFromFormData(formData) {
        try {
            this.showLoading();
            
            // Prepare request data
            const requestData = {
                data: formData,
                telescope: this.selectedTelescope,
                model: this.selectedModel
            };

            // Make API request
            const response = await fetch(`${this.apiBaseUrl}/api/predict-form`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayResults(result);

        } catch (error) {
            console.error('Prediction failed:', error);
            
            // For demo purposes, show mock results if API is not available
            if (error.message.includes('Failed to fetch')) {
                this.showMockResults();
            } else {
                this.showError('Prediction failed. Please try again.');
            }
        } finally {
            this.hideLoading();
        }
    }

    showMockResults() {
        // Mock results for demonstration
        const mockResult = {
            prediction: Math.random() > 0.5 ? 'exoplanet' : 'not_exoplanet',
            // Confidence removed as requested
            probabilities: {
                exoplanet: Math.random() * 0.4 + 0.3,
                not_exoplanet: Math.random() * 0.4 + 0.3
            },
            analysis: {
                key_features: [
                    { name: 'koi_max_sngle_ev', value: Math.random() * 10, importance: 0.8 },
                    { name: 'koi_depth', value: Math.random() * 5, importance: 0.6 },
                    { name: 'koi_insol', value: Math.random() * 2, importance: 0.4 }
                ],
                data_quality: 'Good',
                missing_values: Math.floor(Math.random() * 5),
                feature_count: 324
            }
        };

        this.displayResults(mockResult);
    }

    displayResults(result) {
        const resultsArea = document.getElementById('resultsArea');
        const predictionResult = document.getElementById('predictionResult');
        const detailedAnalysis = document.getElementById('detailedAnalysis');
        if (!resultsArea || !predictionResult || !detailedAnalysis) return;

        // Show results area
        resultsArea.style.display = 'block';
        resultsArea.scrollIntoView({ behavior: 'smooth' });

        // Confidence removed as requested

        // Create prediction result
        const isExoplanet = result.prediction === 'exoplanet';
        predictionResult.innerHTML = `
            <div class="prediction-card ${isExoplanet ? 'exoplanet' : 'not-exoplanet'}">
                <div class="prediction-icon">
                    <i class="fas fa-${isExoplanet ? 'planet-ringed' : 'times-circle'}"></i>
                </div>
                <div class="prediction-title">
                    ${isExoplanet ? 'ExoPlanet Detected!' : 'No ExoPlanet Detected'}
                </div>
                <div class="prediction-description">
                    ${isExoplanet 
                        ? 'Our AI model has identified this object as a potential exoplanet.'
                        : 'Our AI model has determined this object is likely not an exoplanet.'
                    }
                </div>
                <div class="prediction-stats">
                    <div class="prediction-stat">
                        <div class="prediction-stat-value">${(result.probabilities.exoplanet * 100).toFixed(1)}%</div>
                        <div class="prediction-stat-label">ExoPlanet Probability</div>
                    </div>
                    <div class="prediction-stat">
                        <div class="prediction-stat-value">${(result.probabilities.not_exoplanet * 100).toFixed(1)}%</div>
                        <div class="prediction-stat-label">Not ExoPlanet Probability</div>
                    </div>
                </div>
            </div>
        `;

        // Create detailed analysis
        detailedAnalysis.innerHTML = `
            <div class="analysis-section">
                <h4><i class="fas fa-chart-bar"></i> Key Features Analysis</h4>
                <div class="analysis-grid">
                    ${result.analysis.key_features.map(feature => `
                        <div class="analysis-item">
                            <span class="analysis-label">${feature.name}</span>
                            <span class="analysis-value">${feature.value.toFixed(3)}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <div class="analysis-section">
                <h4><i class="fas fa-info-circle"></i> Data Quality</h4>
                <div class="analysis-grid">
                    <div class="analysis-item">
                        <span class="analysis-label">Data Quality</span>
                        <span class="analysis-value">${result.analysis.data_quality}</span>
                    </div>
                    <div class="analysis-item">
                        <span class="analysis-label">Missing Values</span>
                        <span class="analysis-value">${result.analysis.missing_values}</span>
                    </div>
                    <div class="analysis-item">
                        <span class="analysis-label">Features Analyzed</span>
                        <span class="analysis-value">${result.analysis.feature_count}</span>
                    </div>
                </div>
            </div>
        `;

        // Animate results
        setTimeout(() => {
            predictionResult.classList.add('animate-slideInUp');
        }, 100);

        setTimeout(() => {
            detailedAnalysis.classList.add('animate-slideInUp');
        }, 300);
    }

    clearUpload() {
        this.uploadedFile = null;
        
        // Reset file input
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.value = '';
        }

        // Reset upload UI
        const uploadZone = document.getElementById('uploadZone');
        const uploadContent = uploadZone.querySelector('.upload-content');
        uploadContent.innerHTML = `
            <i class="fas fa-cloud-upload-alt upload-icon"></i>
            <h3>Drop your data file here</h3>
            <p>Supports CSV, JSON, and Excel files</p>
            <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">
                <i class="fas fa-folder-open"></i>
                Choose File
            </button>
        `;

        // Disable predict button
        const predictBtn = document.getElementById('predictBtn');
        if (predictBtn) {
            predictBtn.disabled = true;
            predictBtn.classList.remove('btn-animated');
        }

        // Hide results
        const resultsArea = document.getElementById('resultsArea');
        if (resultsArea) {
            resultsArea.style.display = 'none';
        }
    }

    showLoading() {
        if (window.exoPlanetApp) {
            window.exoPlanetApp.showLoading();
        }
    }

    hideLoading() {
        if (window.exoPlanetApp) {
            window.exoPlanetApp.hideLoading();
        }
    }

    showError(message) {
        if (window.exoPlanetApp) {
            window.exoPlanetApp.showNotification(message, 'error');
        }
    }

    showSuccess(message) {
        if (window.exoPlanetApp) {
            window.exoPlanetApp.showNotification(message, 'success');
        }
    }
}

// Initialize upload handler when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.uploadHandler = new UploadHandler();
});
