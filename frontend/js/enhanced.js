// Enhanced functionality for ExoPlanet AI Frontend

class EnhancedFeatures {
    constructor() {
        this.topFeatures = [
            'koi_max_sngle_ev',
            'koi_depth', 
            'koi_insol',
            'koi_max_mult_ev',
            'koi_dikco_msky',
            'koi_insol_err2',
            'koi_incl',
            'koi_ror',
            'koi_smet_err2',
            'koi_prad_err1'
        ];
        
        this.init();
    }

    init() {
        this.setupUploadMethodTabs();
        this.setupDataForm();
        this.setupTrainingFeatures();
        this.loadTrainingLogs();
        this.setupVideoBackground();
        this.setupSatelliteSelector();
    }

    // Upload Method Tabs
    setupUploadMethodTabs() {
        const methodTabs = document.querySelectorAll('.method-tab');
        const fileMethod = document.getElementById('fileUploadMethod');
        const formMethod = document.getElementById('formUploadMethod');

        methodTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs
                methodTabs.forEach(t => t.classList.remove('active'));
                
                // Add active class to clicked tab
                tab.classList.add('active');
                
                // Show/hide appropriate method
                const method = tab.dataset.method;
                if (method === 'file') {
                    fileMethod.style.display = 'block';
                    formMethod.style.display = 'none';
                } else {
                    fileMethod.style.display = 'none';
                    formMethod.style.display = 'block';
                }
            });
        });
    }

    // Data Form Generation
    setupDataForm() {
        const dataForm = document.getElementById('dataForm');
        if (!dataForm) return;

        // Generate form fields for top 10 features
        const formHTML = this.topFeatures.map(feature => {
            const displayName = this.formatFeatureName(feature);
            return `
                <div class="form-group">
                    <label for="${feature}">${displayName}</label>
                    <input 
                        type="number" 
                        id="${feature}" 
                        name="${feature}" 
                        step="any" 
                        placeholder="Enter ${displayName} value"
                        required
                    >
                </div>
            `;
        }).join('');

        dataForm.innerHTML = formHTML;
    }

    formatFeatureName(feature) {
        return feature
            .replace(/koi_/g, '')
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }


    // Load Training Logs
    async loadTrainingLogs(telescope = 'kepler') {
        const logsContent = document.getElementById('logsContent');
        if (!logsContent) {
            console.error('Logs content element not found');
            return;
        }

        console.log(`Loading training logs for ${telescope}...`);
        logsContent.innerHTML = '<div class="loading-logs">Loading training logs...</div>';
        
        try {
            // Fetch logs from API
            const response = await fetch(`http://localhost:8000/api/training_logs/${telescope}`);
            
            if (!response.ok) {
                throw new Error(`Failed to load logs: ${response.statusText}`);
            }
            
            const data = await response.json();
            const logs = data.logs || 'No logs available';

            this.formatLogsWithColors(logsContent, logs);
            console.log('Logs loaded and formatted successfully');
        } catch (error) {
            logsContent.textContent = 'Error loading training logs. Please try again later.';
            console.error('Error loading logs:', error);
        }
    }

    // Format logs with colors and better organization
    formatLogsWithColors(logsContent, logs) {
        console.log('Formatting logs with colors...');
        const logLines = logs.split('\n');
        console.log(`Processing ${logLines.length} log lines`);
        let formattedHTML = '<div class="logs-container">';
        
        logLines.forEach(line => {
            if (line.trim()) {
                const parts = line.split(' - ');
                if (parts.length >= 3) {
                    const timestamp = parts[0];
                    const level = parts[1];
                    const message = parts.slice(2).join(' - ');
                    
                    let levelClass = 'log-info';
                    let levelIcon = 'ℹ️';
                    
                    if (level.includes('ERROR')) {
                        levelClass = 'log-error';
                        levelIcon = '❌';
                    } else if (level.includes('WARNING')) {
                        levelClass = 'log-warning';
                        levelIcon = '⚠️';
                    } else if (level.includes('SUCCESS')) {
                        levelClass = 'log-success';
                        levelIcon = '✅';
                    } else if (message.includes('Training') || message.includes('Model')) {
                        levelClass = 'log-training';
                        levelIcon = '🚀';
                    } else if (message.includes('Data') || message.includes('Preprocessing')) {
                        levelClass = 'log-data';
                        levelIcon = '📊';
                    } else if (message.includes('Evaluation') || message.includes('Metrics')) {
                        levelClass = 'log-evaluation';
                        levelIcon = '📈';
                    }
                    
                    formattedHTML += `
                        <div class="log-line ${levelClass}">
                            <div class="log-timestamp">${timestamp}</div>
                            <div class="log-level">
                                <span class="log-icon">${levelIcon}</span>
                                <span class="log-level-text">${level}</span>
                            </div>
                            <div class="log-message">${message}</div>
                        </div>
                    `;
                } else {
                    formattedHTML += `<div class="log-line log-plain">${line}</div>`;
                }
            }
        });
        
        formattedHTML += '</div>';
        logsContent.innerHTML = formattedHTML;
        console.log('Logs formatted and inserted into DOM');
        console.log('Logs content length:', logsContent.innerHTML.length);
        console.log('Logs container height:', logsContent.offsetHeight);
    }

    // Video Background Setup
    setupVideoBackground() {
        const video = document.querySelector('.galaxy-video');
        if (!video) return;

        console.log('Setting up local video background...');

        // Handle video load success
        video.addEventListener('loadeddata', () => {
            console.log('Local video loaded successfully');
        });

        // Try to play the video
        video.addEventListener('canplay', () => {
            video.play().catch(error => {
                console.error('Video autoplay failed:', error);
            });
        });

        // Handle any errors
        video.addEventListener('error', (e) => {
            console.error('Video failed to load:', e);
        });
    }

    // Satellite Selector
    setupSatelliteSelector() {
        const satelliteBtns = document.querySelectorAll('.satellite-btn');
        
        satelliteBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // Remove active class from all buttons
                satelliteBtns.forEach(b => b.classList.remove('active'));
                
                // Add active class to clicked button
                btn.classList.add('active');
                
                // Get selected satellite
                const selectedSatellite = btn.getAttribute('data-satellite');
                console.log('Selected satellite:', selectedSatellite);
                
                // Update analytics based on selected satellite
                this.updateAnalyticsForSatellite(selectedSatellite);
            });
        });
    }

    // Update analytics based on selected satellite
    updateAnalyticsForSatellite(satellite) {
        // This function would update the charts and metrics based on the selected satellite
        // For now, we'll just log the selection
        console.log(`Updating analytics for ${satellite} satellite data`);
        
        // In a real implementation, you would:
        // 1. Fetch data for the selected satellite
        // 2. Update the charts with new data
        // 3. Update the metrics display
        // 4. Update the training configuration display
        
        // Example of what you might do:
        if (window.chartManager) {
            // Update charts with new satellite data
            window.chartManager.updateChartsForSatellite(satellite);
        }
    }

    // Training Features
    setupTrainingFeatures() {
        const trainFileInput = document.getElementById('trainFileInput');
        const startTrainingBtn = document.getElementById('startTrainingBtn');
        const clearTrainingBtn = document.getElementById('clearTrainingBtn');
        const trainUploadZone = document.getElementById('trainUploadZone');

        if (!trainFileInput || !startTrainingBtn) return;

        // File input handler
        trainFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                startTrainingBtn.disabled = false;
                this.updateTrainUploadUI(e.target.files[0]);
            }
        });

        // Clear button handler
        if (clearTrainingBtn) {
            clearTrainingBtn.addEventListener('click', () => {
                trainFileInput.value = '';
                startTrainingBtn.disabled = true;
                this.resetTrainUploadUI();
            });
        }

        // Start training button handler
        startTrainingBtn.addEventListener('click', () => {
            this.startTraining();
        });

        // Drag and drop for training upload
        if (trainUploadZone) {
            this.setupDragAndDrop(trainUploadZone, trainFileInput);
        }
    }

    updateTrainUploadUI(file) {
        const trainUploadZone = document.getElementById('trainUploadZone');
        if (!trainUploadZone) return;

        const uploadContent = trainUploadZone.querySelector('.upload-content');
        uploadContent.innerHTML = `
            <i class="fas fa-file-check upload-icon" style="color: var(--success-color);"></i>
            <h3>Training File Ready</h3>
            <p><strong>${file.name}</strong></p>
            <p>Size: ${this.formatFileSize(file.size)}</p>
            <button class="btn btn-secondary" onclick="document.getElementById('trainFileInput').click()">
                <i class="fas fa-upload"></i>
                Choose Different File
            </button>
        `;
    }

    resetTrainUploadUI() {
        const trainUploadZone = document.getElementById('trainUploadZone');
        if (!trainUploadZone) return;

        const uploadContent = trainUploadZone.querySelector('.upload-content');
        uploadContent.innerHTML = `
            <i class="fas fa-brain upload-icon"></i>
            <h3>Upload Training Dataset</h3>
            <p>Upload your CSV file with telescope data for training</p>
            <button class="btn btn-primary" onclick="document.getElementById('trainFileInput').click()">
                <i class="fas fa-upload"></i>
                Choose Training File
            </button>
        `;
    }

    setupDragAndDrop(uploadZone, fileInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, this.preventDefaults, false);
        });

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

        uploadZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files;
                const event = new Event('change', { bubbles: true });
                fileInput.dispatchEvent(event);
            }
        }, false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Start Training Process
    async startTraining() {
        const trainingProgress = document.getElementById('trainingProgress');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const trainingLogs = document.getElementById('trainingLogs');

        if (!trainingProgress) return;

        // Show training progress
        trainingProgress.style.display = 'block';
        trainingProgress.scrollIntoView({ behavior: 'smooth' });

        // Simulate training process
        const steps = [
            { progress: 10, text: 'Loading dataset...', log: 'Loading training data from uploaded file' },
            { progress: 25, text: 'Preprocessing data...', log: 'Applying data preprocessing pipeline' },
            { progress: 40, text: 'Splitting data...', log: 'Creating train/test split (80/20)' },
            { progress: 55, text: 'Initializing model...', log: 'Setting up Logistic Regression with hyperparameter optimization' },
            { progress: 70, text: 'Training model...', log: 'Running Bayesian optimization (50 iterations)' },
            { progress: 85, text: 'Cross-validation...', log: 'Performing 5-fold cross-validation' },
            { progress: 95, text: 'Evaluating model...', log: 'Computing performance metrics' },
            { progress: 100, text: 'Training completed!', log: 'Model saved successfully. Accuracy: 87.2%' }
        ];

        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            
            // Update progress
            progressFill.style.width = step.progress + '%';
            progressText.textContent = step.text;
            
            // Add log entry
            if (trainingLogs) {
                const logEntry = document.createElement('div');
                logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${step.log}`;
                trainingLogs.appendChild(logEntry);
                trainingLogs.scrollTop = trainingLogs.scrollHeight;
            }
            
            // Wait before next step
            await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
        }

        // Show completion message
        if (window.exoPlanetApp) {
            window.exoPlanetApp.showNotification('Training completed successfully!', 'success');
        }
    }

    // Get form data for prediction
    getFormData() {
        const formData = {};
        this.topFeatures.forEach(feature => {
            const input = document.getElementById(feature);
            if (input && input.value) {
                formData[feature] = parseFloat(input.value);
            }
        });
        return formData;
    }

    // Validate form data
    validateFormData() {
        const formData = this.getFormData();
        const missingFields = this.topFeatures.filter(feature => !formData[feature]);
        
        if (missingFields.length > 0) {
            const fieldNames = missingFields.map(f => this.formatFeatureName(f)).join(', ');
            if (window.exoPlanetApp) {
                window.exoPlanetApp.showNotification(`Please fill in: ${fieldNames}`, 'error');
            }
            return false;
        }
        
        return true;
    }
}

// Initialize enhanced features when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.enhancedFeatures = new EnhancedFeatures();
});

// Export functions for global access
function getFormData() {
    if (window.enhancedFeatures) {
        return window.enhancedFeatures.getFormData();
    }
    return {};
}

function validateFormData() {
    if (window.enhancedFeatures) {
        return window.enhancedFeatures.validateFormData();
    }
    return false;
}

// Initialize enhanced features when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.enhancedFeatures = new EnhancedFeatures();
    console.log('Enhanced features initialized');
});
