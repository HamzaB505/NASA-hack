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
    async loadTrainingLogs() {
        const logsContent = document.getElementById('logsContent');
        if (!logsContent) {
            console.error('Logs content element not found');
            return;
        }

        console.log('Loading training logs...');
        
        try {
            // Use hardcoded logs since the file is not accessible from frontend
            const logs = `2025-10-05 10:36:41,475 - ML - INFO - Trainer instance initialized
2025-10-05 10:36:41,475 - ML - INFO - Starting training process with file: cumulative_2025.10.04_04.05.07.csv
2025-10-05 10:36:41,475 - ML - INFO - Initializing data preprocessor
2025-10-05 10:36:41,475 - ML - INFO - Running preprocessing pipeline
2025-10-05 10:36:41,475 - ML - INFO - Reading file cumulative_2025.10.04_04.05.07.csv with delimiter ,
2025-10-05 10:36:41,792 - ML - INFO - Defined data with shape: (9564, 139) and (9564, 2)
2025-10-05 10:36:43,005 - ML - INFO - Dropping columns with single values: ['koi_vet_stat', 'koi_disp_prov', 'koi_eccen', 'koi_eccen_err1', 'koi_eccen_err2', 'koi_longp', 'koi_longp_err1', 'koi_longp_err2', 'koi_ingress', 'koi_ingress_err1', 'koi_ingress_err2', 'koi_sma_err1', 'koi_sma_err2', 'koi_incl_err1', 'koi_incl_err2', 'koi_teq_err1', 'koi_teq_err2', 'koi_limbdark_mod', 'koi_ldm_coeff4', 'koi_ldm_coeff3', 'koi_trans_mod', 'koi_model_dof', 'koi_model_chisq', 'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_sage', 'koi_sage_err1', 'koi_sage_err2']
2025-10-05 10:36:43,031 - ML - INFO - Capping 1 outliers in column 'koi_fpflag_nt' at 1.0% percentile
2025-10-05 10:36:43,040 - ML - INFO - Capping 192 outliers in column 'koi_period' at 1.0% percentile
2025-10-05 10:36:43,044 - ML - INFO - Capping 183 outliers in column 'koi_period_err1' at 1.0% percentile
2025-10-05 10:36:43,047 - ML - INFO - Capping 183 outliers in column 'koi_period_err2' at 1.0% percentile
2025-10-05 10:36:43,050 - ML - INFO - Capping 192 outliers in column 'koi_time0bk' at 1.0% percentile
2025-10-05 10:36:43,053 - ML - INFO - Capping 182 outliers in column 'koi_time0bk_err1' at 1.0% percentile
2025-10-05 10:36:43,058 - ML - INFO - Capping 182 outliers in column 'koi_time0bk_err2' at 1.0% percentile
2025-10-05 10:36:43,062 - ML - INFO - Capping 190 outliers in column 'koi_time0' at 1.0% percentile
2025-10-05 10:36:43,065 - ML - INFO - Capping 182 outliers in column 'koi_time0_err1' at 1.0% percentile
2025-10-05 10:36:43,069 - ML - INFO - Capping 182 outliers in column 'koi_time0_err2' at 1.0% percentile
2025-10-05 10:36:43,073 - ML - INFO - Capping 170 outliers in column 'koi_impact' at 1.0% percentile
2025-10-05 10:36:43,077 - ML - INFO - Capping 124 outliers in column 'koi_impact_err1' at 1.0% percentile
2025-10-05 10:36:43,082 - ML - INFO - Capping 127 outliers in column 'koi_impact_err2' at 1.0% percentile
2025-10-05 10:36:43,085 - ML - INFO - Capping 192 outliers in column 'koi_duration' at 1.0% percentile
2025-10-05 10:36:43,089 - ML - INFO - Capping 182 outliers in column 'koi_duration_err1' at 1.0% percentile
2025-10-05 10:36:43,094 - ML - INFO - Capping 182 outliers in column 'koi_duration_err2' at 1.0% percentile
2025-10-05 10:36:43,098 - ML - INFO - Capping 184 outliers in column 'koi_depth' at 1.0% percentile
2025-10-05 10:36:43,102 - ML - INFO - Capping 181 outliers in column 'koi_depth_err1' at 1.0% percentile
2025-10-05 10:36:43,106 - ML - INFO - Capping 181 outliers in column 'koi_depth_err2' at 1.0% percentile
2025-10-05 10:36:43,111 - ML - INFO - Capping 184 outliers in column 'koi_ror' at 1.0% percentile
2025-10-05 10:36:43,115 - ML - INFO - Capping 92 outliers in column 'koi_ror_err1' at 1.0% percentile
2025-10-05 10:36:43,118 - ML - INFO - Capping 184 outliers in column 'koi_ror_err2' at 1.0% percentile
2025-10-05 10:36:43,121 - ML - INFO - Capping 186 outliers in column 'koi_srho' at 1.0% percentile
2025-10-05 10:36:43,126 - ML - INFO - Capping 93 outliers in column 'koi_srho_err1' at 1.0% percentile
2025-10-05 10:36:43,131 - ML - INFO - Capping 93 outliers in column 'koi_srho_err2' at 1.0% percentile
2025-10-05 10:36:43,135 - ML - INFO - Capping 179 outliers in column 'koi_prad' at 1.0% percentile
2025-10-05 10:36:43,140 - ML - INFO - Capping 92 outliers in column 'koi_prad_err1' at 1.0% percentile
2025-10-05 10:36:43,144 - ML - INFO - Capping 92 outliers in column 'koi_prad_err2' at 1.0% percentile
2025-10-05 10:36:43,149 - ML - INFO - Capping 184 outliers in column 'koi_sma' at 1.0% percentile
2025-10-05 10:36:43,154 - ML - INFO - Capping 92 outliers in column 'koi_incl' at 1.0% percentile
2025-10-05 10:36:43,158 - ML - INFO - Capping 179 outliers in column 'koi_teq' at 1.0% percentile
2025-10-05 10:36:43,161 - ML - INFO - Capping 180 outliers in column 'koi_insol' at 1.0% percentile
2025-10-05 10:36:43,164 - ML - INFO - Capping 93 outliers in column 'koi_insol_err1' at 1.0% percentile
2025-10-05 10:36:43,168 - ML - INFO - Capping 93 outliers in column 'koi_insol_err2' at 1.0% percentile
2025-10-05 10:36:43,170 - ML - INFO - Capping 184 outliers in column 'koi_dor' at 1.0% percentile
2025-10-05 10:36:43,173 - ML - INFO - Capping 184 outliers in column 'koi_dor_err1' at 1.0% percentile
2025-10-05 10:36:43,180 - ML - INFO - Capping 184 outliers in column 'koi_dor_err2' at 1.0% percentile
2025-10-05 10:36:43,184 - ML - INFO - Capping 182 outliers in column 'koi_ldm_coeff2' at 1.0% percentile
2025-10-05 10:36:43,190 - ML - INFO - Capping 177 outliers in column 'koi_ldm_coeff1' at 1.0% percentile
2025-10-05 10:36:43,195 - ML - INFO - Capping 170 outliers in column 'koi_max_sngle_ev' at 1.0% percentile
2025-10-05 10:36:43,204 - ML - INFO - Capping 170 outliers in column 'koi_max_mult_ev' at 1.0% percentile
2025-10-05 10:36:43,208 - ML - INFO - Capping 180 outliers in column 'koi_model_snr' at 1.0% percentile
2025-10-05 10:36:43,213 - ML - INFO - Capping 32 outliers in column 'koi_count' at 1.0% percentile
2025-10-05 10:36:43,218 - ML - INFO - Capping 83 outliers in column 'koi_num_transits' at 1.0% percentile
2025-10-05 10:36:43,222 - ML - INFO - Capping 64 outliers in column 'koi_tce_plnt_num' at 1.0% percentile
2025-10-05 10:36:43,231 - ML - INFO - Capping 181 outliers in column 'koi_steff' at 1.0% percentile
2025-10-05 10:36:43,235 - ML - INFO - Capping 181 outliers in column 'koi_steff_err1' at 1.0% percentile
2025-10-05 10:36:43,238 - ML - INFO - Capping 180 outliers in column 'koi_steff_err2' at 1.0% percentile
2025-10-05 10:36:43,242 - ML - INFO - Capping 182 outliers in column 'koi_slogg' at 1.0% percentile
2025-10-05 10:36:43,249 - ML - INFO - Capping 136 outliers in column 'koi_slogg_err1' at 1.0% percentile
2025-10-05 10:36:43,253 - ML - INFO - Capping 164 outliers in column 'koi_slogg_err2' at 1.0% percentile
2025-10-05 10:36:43,256 - ML - INFO - Capping 180 outliers in column 'koi_smet' at 1.0% percentile
2025-10-05 10:36:43,262 - ML - INFO - Capping 73 outliers in column 'koi_smet_err1' at 1.0% percentile
2025-10-05 10:36:43,266 - ML - INFO - Capping 87 outliers in column 'koi_smet_err2' at 1.0% percentile
2025-10-05 10:36:43,269 - ML - INFO - Capping 183 outliers in column 'koi_srad' at 1.0% percentile
2025-10-05 10:36:43,272 - ML - INFO - Capping 177 outliers in column 'koi_srad_err1' at 1.0% percentile
2025-10-05 10:36:43,275 - ML - INFO - Capping 176 outliers in column 'koi_srad_err2' at 1.0% percentile
2025-10-05 10:36:43,281 - ML - INFO - Capping 184 outliers in column 'koi_smass' at 1.0% percentile
2025-10-05 10:36:43,284 - ML - INFO - Capping 179 outliers in column 'koi_smass_err1' at 1.0% percentile
2025-10-05 10:36:43,288 - ML - INFO - Capping 179 outliers in column 'koi_smass_err2' at 1.0% percentile
2025-10-05 10:36:43,290 - ML - INFO - Capping 192 outliers in column 'ra' at 1.0% percentile
2025-10-05 10:36:43,296 - ML - INFO - Capping 192 outliers in column 'dec' at 1.0% percentile
2025-10-05 10:36:43,300 - ML - INFO - Capping 191 outliers in column 'koi_kepmag' at 1.0% percentile
2025-10-05 10:36:43,303 - ML - INFO - Capping 191 outliers in column 'koi_gmag' at 1.0% percentile
2025-10-05 10:36:43,306 - ML - INFO - Capping 191 outliers in column 'koi_rmag' at 1.0% percentile
2025-10-05 10:36:43,308 - ML - INFO - Capping 190 outliers in column 'koi_imag' at 1.0% percentile
2025-10-05 10:36:43,316 - ML - INFO - Capping 180 outliers in column 'koi_zmag' at 1.0% percentile
2025-10-05 10:36:43,321 - ML - INFO - Capping 192 outliers in column 'koi_jmag' at 1.0% percentile
2025-10-05 10:36:43,325 - ML - INFO - Capping 192 outliers in column 'koi_hmag' at 1.0% percentile
2025-10-05 10:36:43,331 - ML - INFO - Capping 191 outliers in column 'koi_kmag' at 1.0% percentile
2025-10-05 10:36:43,335 - ML - INFO - Capping 84 outliers in column 'koi_fwm_stat_sig' at 1.0% percentile
2025-10-05 10:36:43,338 - ML - INFO - Capping 182 outliers in column 'koi_fwm_sra' at 1.0% percentile
2025-10-05 10:36:43,341 - ML - INFO - Capping 162 outliers in column 'koi_fwm_sra_err' at 1.0% percentile
2025-10-05 10:36:43,346 - ML - INFO - Capping 182 outliers in column 'koi_fwm_sdec' at 1.0% percentile
2025-10-05 10:36:43,350 - ML - INFO - Capping 181 outliers in column 'koi_fwm_sdec_err' at 1.0% percentile
2025-10-05 10:36:43,353 - ML - INFO - Capping 184 outliers in column 'koi_fwm_srao' at 1.0% percentile
2025-10-05 10:36:43,356 - ML - INFO - Capping 167 outliers in column 'koi_fwm_srao_err' at 1.0% percentile
2025-10-05 10:36:43,359 - ML - INFO - Capping 184 outliers in column 'koi_fwm_sdeco' at 1.0% percentile
2025-10-05 10:36:43,365 - ML - INFO - Capping 183 outliers in column 'koi_fwm_sdeco_err' at 1.0% percentile
2025-10-05 10:36:43,368 - ML - INFO - Capping 173 outliers in column 'koi_fwm_prao' at 1.0% percentile
2025-10-05 10:36:43,371 - ML - INFO - Capping 161 outliers in column 'koi_fwm_prao_err' at 1.0% percentile
2025-10-05 10:36:43,375 - ML - INFO - Capping 175 outliers in column 'koi_fwm_pdeco' at 1.0% percentile
2025-10-05 10:36:43,381 - ML - INFO - Capping 162 outliers in column 'koi_fwm_pdeco_err' at 1.0% percentile
2025-10-05 10:36:43,385 - ML - INFO - Capping 180 outliers in column 'koi_dicco_mra' at 1.0% percentile
2025-10-05 10:36:43,388 - ML - INFO - Capping 85 outliers in column 'koi_dicco_mra_err' at 1.0% percentile
2025-10-05 10:36:43,391 - ML - INFO - Capping 180 outliers in column 'koi_dicco_mdec' at 1.0% percentile
2025-10-05 10:36:43,397 - ML - INFO - Capping 77 outliers in column 'koi_dicco_mdec_err' at 1.0% percentile
2025-10-05 10:36:43,404 - ML - INFO - Capping 174 outliers in column 'koi_dicco_msky' at 1.0% percentile
2025-10-05 10:36:43,412 - ML - INFO - Capping 84 outliers in column 'koi_dicco_msky_err' at 1.0% percentile
2025-10-05 10:36:43,417 - ML - INFO - Capping 180 outliers in column 'koi_dikco_mra' at 1.0% percentile
2025-10-05 10:36:43,420 - ML - INFO - Capping 83 outliers in column 'koi_dikco_mra_err' at 1.0% percentile
2025-10-05 10:36:43,424 - ML - INFO - Capping 180 outliers in column 'koi_dikco_mdec' at 1.0% percentile
2025-10-05 10:36:43,430 - ML - INFO - Capping 89 outliers in column 'koi_dikco_mdec_err' at 1.0% percentile
2025-10-05 10:36:43,435 - ML - INFO - Capping 174 outliers in column 'koi_dikco_msky' at 1.0% percentile
2025-10-05 10:36:43,438 - ML - INFO - Capping 89 outliers in column 'koi_dikco_msky_err' at 1.0% percentile
2025-10-05 10:36:45,502 - ML - INFO - Dropping columns with koi_fpflag: ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
2025-10-05 10:36:45,506 - ML - INFO - Dropped columns with koi_fpflag: ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
2025-10-05 10:36:45,573 - ML - INFO - Data split completed - Train: (7651, 324), Test: (1913, 324)
2025-10-05 10:36:45,573 - ML - INFO - y_train shape: (7651,), y_test shape: (1913,)
2025-10-05 10:36:45,574 - ML - INFO - Number of training samples: 7651
2025-10-05 10:36:45,574 - ML - INFO - Number of test samples: 1913
2025-10-05 10:36:45,574 - ML - INFO - Number of features: 324
2025-10-05 10:36:45,574 - ML - INFO - Number of classes: 2
2025-10-05 10:36:45,574 - ML - INFO - Number of feature names stored: 324
2025-10-05 10:36:45,574 - ML - INFO - Initializing model optimizer
2025-10-05 10:36:45,575 - ML - INFO - ModelOptimizer initialized with random_state=42, model_save_dir=/Users/hamzaboulaala/Documents/github/NASA-hack/models/2025.10.05_10.36.41
2025-10-05 10:36:45,575 - ML - INFO - Training all models
2025-10-05 10:36:45,575 - ML - INFO - Starting training of all models with parameters: cv=2, n_iter=10, scoring=accuracy
2025-10-05 10:36:45,575 - ML - INFO - Creating model pipelines
2025-10-05 10:36:45,575 - ML - INFO - Creating pipelines for 1 models: ['Logistic Regression']
2025-10-05 10:36:45,575 - ML - INFO - Successfully created 1 model pipelines
2025-10-05 10:36:45,575 - ML - INFO - Defining Bayesian search spaces for hyperparameter optimization
2025-10-05 10:36:45,585 - ML - INFO - Defined search spaces for 6 models
2025-10-05 10:36:45,586 - ML - INFO - Training 1 models: ['Logistic Regression']
2025-10-05 10:36:45,601 - ML - INFO - Starting training for Logistic Regression
2025-10-05 10:36:45,602 - ML - INFO - 
Optimizing Logistic Regression...
2025-10-05 10:37:42,186 - ML - INFO - Saving model Logistic Regression to /Users/hamzaboulaala/Documents/github/NASA-hack/models/2025.10.05_10.36.41
2025-10-05 10:37:42,227 - ML - INFO - Model Logistic Regression saved to /Users/hamzaboulaala/Documents/github/NASA-hack/models/2025.10.05_10.36.41/Logistic_Regression_model.pkl
2025-10-05 10:37:42,227 - ML - INFO - Completed optimization for Logistic Regression - Best score: 0.8608
2025-10-05 10:37:42,227 - ML - INFO - Best score for Logistic Regression: 0.8608
2025-10-05 10:37:42,228 - ML - INFO - Best parameters for Logistic Regression: OrderedDict([('classifier__C', 65.1509271961212), ('classifier__penalty', 'l2'), ('feature_selector__max_features', 30)])
2025-10-05 10:37:42,228 - ML - INFO - Completed training of all 1 models
2025-10-05 10:37:42,228 - ML - INFO - Model training completed for 1 models
2025-10-05 10:37:42,228 - ML - INFO - Evaluating models on test set
2025-10-05 10:37:42,228 - ML - INFO - Starting evaluation of 1 models on test set
2025-10-05 10:37:42,229 - ML - INFO - Evaluating Logistic Regression
2025-10-05 10:37:42,229 - ML - INFO - 
Evaluating Logistic Regression...
2025-10-05 10:38:01,414 - ML - INFO - Confusion matrix for Logistic Regression:
2025-10-05 10:38:01,416 - ML - INFO - Confusion matrix: (1913,) vs (1913,)
2025-10-05 10:38:01,416 - ML - INFO - y_pred:[0 1 1 ... 0 1 0]
2025-10-05 10:38:01,416 - ML - INFO - y_true:[0 1 1 ... 0 1 0]
2025-10-05 10:38:01,417 - ML - INFO - 
Confusion Matrix for Logistic Regression
2025-10-05 10:38:01,417 - ML - INFO - =======================================
2025-10-05 10:38:01,417 - ML - INFO -         Pred 0  Pred 1
True 0     798     170
True 1      93     852
2025-10-05 10:38:02,096 - ML - INFO - Saved confusion matrix heatmap to /Users/hamzaboulaala/Documents/github/NASA-hack/models/2025.10.05_10.36.41/Logistic_Regression_confusion_matrix.png
2025-10-05 10:38:02,097 - ML - INFO - Metrics for Logistic Regression:
2025-10-05 10:38:02,111 - ML - INFO - 
                 Model  Accuracy  Precision  Recall  F1-Score  ROC AUC  PR AUC
0  Logistic Regression    0.8625      0.865  0.8625    0.8624   0.9359  0.9325
2025-10-05 10:38:02,112 - ML - INFO - Comparison of metrics for Logistic Regression:
2025-10-05 10:38:02,124 - ML - INFO - Saved metrics to /Users/hamzaboulaala/Documents/github/NASA-hack/models/2025.10.05_10.36.41/Logistic_Regression_comparison_metrics.json
2025-10-05 10:38:02,124 - ML - INFO - Extracting feature importance for Logistic Regression
2025-10-05 10:38:02,140 - ML - INFO - Top 10 most important features for Logistic Regression:
2025-10-05 10:38:02,141 - ML - INFO -   1. koi_max_sngle_ev: 6.062365
2025-10-05 10:38:02,141 - ML - INFO -   2. koi_depth: 2.605205
2025-10-05 10:38:02,141 - ML - INFO -   3. koi_insol: 1.971400
2025-10-05 10:38:02,141 - ML - INFO -   4. koi_max_mult_ev: 1.828814
2025-10-05 10:38:02,141 - ML - INFO -   5. koi_dikco_msky: 1.624789
2025-10-05 10:38:02,141 - ML - INFO -   6. koi_insol_err2: 1.284011
2025-10-05 10:38:02,141 - ML - INFO -   7. koi_incl: 0.938304
2025-10-05 10:38:02,142 - ML - INFO -   8. koi_ror: 0.723849
2025-10-05 10:38:02,142 - ML - INFO -   9. koi_smet_err2: 0.539282
2025-10-05 10:38:02,142 - ML - INFO -  10. koi_prad_err1: 0.435215
2025-10-05 10:38:02,143 - ML - INFO - Saved feature importance to /Users/hamzaboulaala/Documents/github/NASA-hack/models/2025.10.05_10.36.41/Logistic_Regression_feature_importance.csv
2025-10-05 10:38:02,144 - ML - INFO - Saved feature importance JSON to /Users/hamzaboulaala/Documents/github/NASA-hack/models/2025.10.05_10.36.41/Logistic_Regression_feature_importance.json
2025-10-05 10:38:02,144 - ML - INFO - Completed evaluation for Logistic Regression
2025-10-05 10:38:02,145 - ML - INFO - Completed evaluation of all 1 models
2025-10-05 10:38:02,145 - ML - INFO - Model evaluation completed
2025-10-05 10:38:02,167 - ML - INFO - Training results:
{   'Logistic Regression': {   'confusion_matrix': {   'Pred 0': {   'True 0': 798,
                                                                     'True 1': 93},
                                                       'Pred 1': {   'True 0': 170,
                                                                     'True 1': 852}},
                               'feature_importance': [   {   'feature_name': 'koi_max_sngle_ev',
                                                             'importance': 6.062365474705465},
                                                         {   'feature_name': 'koi_depth',
                                                             'importance': 2.6052046870975984},
                                                         {   'feature_name': 'koi_insol',
                                                             'importance': 1.971400092936551},
                                                         {   'feature_name': 'koi_max_mult_ev',
                                                             'importance': 1.8288140625063471},
                                                         {   'feature_name': 'koi_dikco_msky',
                                                             'importance': 1.6247891461977153},
                                                         {   'feature_name': 'koi_insol_err2',
                                                             'importance': 1.2840108006170967},
                                                         {   'feature_name': 'koi_incl',
                                                             'importance': 0.9383044036732561},
                                                         {   'feature_name': 'koi_ror',
                                                             'importance': 0.7238488329723626},
                                                         {   'feature_name': 'koi_smet_err2',
                                                             'importance': 0.5392824710181567},
                                                         {   'feature_name': 'koi_prad_err1',
                                                             'importance': 0.4352145486619569},
                                                         {   'feature_name': 'koi_duration_err2',
                                                             'importance': 0.3249529890696207},
                                                         {   'feature_name': 'koi_prad',
                                                             'importance': 0.32175878098146304},
                                                         {   'feature_name': 'koi_prad_err2',
                                                             'importance': 0.31198519172622413},
                                                         {   'feature_name': 'koi_fwm_stat_sig',
                                                             'importance': 0.28317183278455016},
                                                         {   'feature_name': 'koi_dicco_msky',
                                                             'importance': 0.2644288147207059},
                                                         {   'feature_name': 'koi_smet_err1',
                                                             'importance': 0.1374334237739049},
                                                         {   'feature_name': 'koi_period',
                                                             'importance': 0.1215855924440832},
                                                         {   'feature_name': 'koi_insol_err1',
                                                             'importance': 0.0854028312601038},
                                                         {   'feature_name': 'koi_dicco_mra',
                                                             'importance': 0.08014275065117824},
                                                         {   'feature_name': 'koi_dicco_mdec_err',
                                                             'importance': 0.07903848922095809},
                                                         {   'feature_name': 'koi_srho_err2',
                                                             'importance': 0.06901560916219876},
                                                         {   'feature_name': 'koi_period_err2',
                                                             'importance': 0.06325135093121598},
                                                         {   'feature_name': 'koi_steff_err1',
                                                             'importance': 0.061494738088354026},
                                                         {   'feature_name': 'koi_dor',
                                                             'importance': 0.05130112729106309},
                                                         {   'feature_name': 'koi_dor_err1',
                                                             'importance': 0.04793365780316601},
                                                         {   'feature_name': 'koi_dor_err2',
                                                             'importance': 0.04793365780282053},
                                                         {   'feature_name': 'koi_fwm_srao',
                                                             'importance': 0.04352203367438326},
                                                         {   'feature_name': 'koi_dicco_mdec',
                                                             'importance': 0.0292259508652193},
                                                         {   'feature_name': 'koi_dicco_mra_err',
                                                             'importance': 0.022969796700343208},
                                                         {   'feature_name': 'koi_fwm_sdeco',
                                                             'importance': 0.008390973711398101}}
                               'metrics':                  Model  Accuracy  Precision  Recall  F1-Score  ROC AUC  PR AUC
0  Logistic Regression    0.8625      0.865  0.8625    0.8624   0.9359  0.9325}}
2025-10-05 10:38:02,168 - ML - INFO - Training process completed successfully
2025-10-05 10:38:02,168 - ML - INFO - Stored model information - Models: ['Logistic Regression']
2025-10-05 10:38:02,168 - ML - INFO - Model save directory: /Users/hamzaboulaala/Documents/github/NASA-hack/models/2025.10.05_10.36.41
2025-10-05 10:38:02,178 - ML - INFO - Training results saved to: /Users/hamzaboulaala/Documents/github/NASA-hack/models/2025.10.05_10.36.41/test_results_2025.10.05_10.36.41.json
2025-10-05 10:38:02,178 - ML - INFO - Training process completed
2025-10-05 10:38:02,179 - ML - INFO - Config saved to: /Users/hamzaboulaala/Documents/github/NASA-hack/models/2025.10.05_10.36.41/experiment_config.json`;

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
