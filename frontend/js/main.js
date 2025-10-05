// Main JavaScript file for ExoPlanet AI Frontend

class ExoPlanetApp {
    constructor() {
        this.currentSection = 'home';
        this.selectedTelescope = 'kepler';
        this.uploadedFile = null;
        this.apiBaseUrl = 'http://localhost:8000'; // FastAPI backend URL
        
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupTelescopeSelection();
        this.setupScrollAnimations();
        this.setupParticleEffects();
        this.loadModelMetrics();
        this.loadRecentDiscoveries();
        this.setupCounterAnimations();
    }

    // Navigation Management
    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        const navToggle = document.querySelector('.nav-toggle');
        const navMenu = document.querySelector('.nav-menu');

        // Handle navigation clicks
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetSection = link.getAttribute('href').substring(1);
                this.navigateToSection(targetSection);
                
                // Close mobile menu
                navMenu.classList.remove('active');
                navToggle.classList.remove('active');
            });
        });

        // Handle mobile menu toggle
        navToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
            navToggle.classList.toggle('active');
        });

        // Handle scroll-based navigation highlighting
        window.addEventListener('scroll', () => {
            this.updateActiveNavLink();
        });
    }

    navigateToSection(sectionId) {
        // Hide current section
        const currentSection = document.querySelector('.section.active');
        if (currentSection) {
            currentSection.classList.remove('active');
        }

        // Show target section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
            this.currentSection = sectionId;
            
            // Update navigation
            this.updateActiveNavLink();
            
            // Scroll to top
            window.scrollTo({ top: 0, behavior: 'smooth' });
            
            // Trigger section-specific animations
            this.triggerSectionAnimations(sectionId);
        }
    }

    updateActiveNavLink() {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${this.currentSection}`) {
                link.classList.add('active');
            }
        });
    }

    // Telescope Selection
    setupTelescopeSelection() {
        const telescopeCards = document.querySelectorAll('.telescope-card');
        
        telescopeCards.forEach(card => {
            card.addEventListener('click', () => {
                // Remove active class from all cards
                telescopeCards.forEach(c => c.classList.remove('selected'));
                
                // Add active class to clicked card
                card.classList.add('selected');
                
                // Update selected telescope
                this.selectedTelescope = card.dataset.telescope;
                
                // Update telescope selector in predict section
                const telescopeSelect = document.getElementById('telescopeSelect');
                if (telescopeSelect) {
                    telescopeSelect.value = this.selectedTelescope;
                }
                
                // Show telescope-specific information
                this.showTelescopeInfo(this.selectedTelescope);
            });
        });

        // Set default selection
        const keplerCard = document.querySelector('[data-telescope="kepler"]');
        if (keplerCard) {
            keplerCard.classList.add('selected');
        }
    }

    showTelescopeInfo(telescope) {
        // Update hero stats based on telescope selection
        const statCards = document.querySelectorAll('.stat-card');
        
        switch (telescope) {
            case 'kepler':
                this.updateStats(statCards, {
                    accuracy: '86.25%',
                    samples: '7651',
                    features: '324'
                });
                break;
            case 'tess':
                this.updateStats(statCards, {
                    accuracy: 'Coming Soon',
                    samples: 'TBD',
                    features: 'TBD'
                });
                break;
            case 'k2':
                this.updateStats(statCards, {
                    accuracy: 'Coming Soon',
                    samples: 'TBD',
                    features: 'TBD'
                });
                break;
        }
    }

    updateStats(statCards, stats) {
        if (statCards.length >= 3) {
            statCards[0].querySelector('.stat-number').textContent = stats.accuracy;
            statCards[1].querySelector('.stat-number').textContent = stats.samples;
            statCards[2].querySelector('.stat-number').textContent = stats.features;
        }
    }

    // Scroll Animations
    setupScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                }
            });
        }, observerOptions);

        // Observe elements for scroll animations
        const animatedElements = document.querySelectorAll('.analytics-card, .telescope-card, .stat-card, .tech-item');
        animatedElements.forEach(el => {
            el.classList.add('scroll-reveal');
            observer.observe(el);
        });
    }

    // Particle Effects
    setupParticleEffects() {
        const galaxyBackground = document.querySelector('.galaxy-background');
        if (!galaxyBackground) return;

        const particlesContainer = document.createElement('div');
        particlesContainer.className = 'particles';
        galaxyBackground.appendChild(particlesContainer);

        // Create floating particles
        for (let i = 0; i < 50; i++) {
            this.createParticle(particlesContainer);
        }
    }

    createParticle(container) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Random positioning and animation delay
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 10 + 's';
        particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
        
        container.appendChild(particle);
    }

    // Load Model Metrics
    async loadModelMetrics() {
        try {
            // In a real implementation, this would fetch from your FastAPI backend
            // For now, we'll use the static data from the JSON files
            const metrics = {
                accuracy: 86.25,
                precision: 86.50,
                recall: 86.25,
                f1Score: 86.24,
                rocAuc: 93.59,
                prAuc: 93.25
            };

            this.animateMetrics(metrics);
        } catch (error) {
            console.error('Error loading model metrics:', error);
        }
    }

    // Load recent discoveries
    async loadRecentDiscoveries() {
        try {
            const response = await fetch('./demo_data.json');
            const data = await response.json();
            
            if (data.recent_discoveries) {
                this.renderDiscoveries(data.recent_discoveries);
            }
        } catch (error) {
            console.error('Error loading discoveries:', error);
        }
    }

    // Render discoveries in the grid
    renderDiscoveries(discoveries) {
        const grid = document.getElementById('discoveriesGrid');
        if (!grid) return;

        grid.innerHTML = discoveries.map(discovery => `
            <div class="discovery-card">
                <div class="discovery-name">${discovery.name}</div>
                <div class="discovery-type">${discovery.type}</div>
                <div class="discovery-metrics">
                    <div class="metric">
                        <div class="metric-value">${discovery.type}</div>
                        <div class="metric-label">Planet Type</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${discovery.status}</div>
                        <div class="metric-label">Status</div>
                    </div>
                </div>
                <div class="discovery-status ${discovery.status.includes('Confirmed') ? 'status-confirmed' : 'status-candidate'}">
                    ${discovery.status}
                </div>
            </div>
        `).join('');
    }

    // Setup counter animations
    setupCounterAnimations() {
        const counters = document.querySelectorAll('.stat-number[data-target]');
        
        const animateCounter = (counter) => {
            const target = parseFloat(counter.dataset.target);
            const duration = 2000; // 2 seconds
            const start = performance.now();
            
            const updateCounter = (currentTime) => {
                const elapsed = currentTime - start;
                const progress = Math.min(elapsed / duration, 1);
                
                // Easing function for smooth animation
                const easeOutQuart = 1 - Math.pow(1 - progress, 4);
                const current = target * easeOutQuart;
                
                if (target < 100) {
                    counter.textContent = current.toFixed(2);
                } else {
                    counter.textContent = Math.floor(current).toLocaleString();
                }
                
                if (progress < 1) {
                    requestAnimationFrame(updateCounter);
                } else {
                    counter.textContent = target < 100 ? target.toFixed(2) : target.toLocaleString();
                }
            };
            
            requestAnimationFrame(updateCounter);
        };

        // Start animations when page loads
        setTimeout(() => {
            counters.forEach(animateCounter);
        }, 500);
    }

    // Live Demo Function
    runLiveDemo() {
        // Show loading overlay
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) overlay.style.display = 'flex';

        // Simulate processing time
        setTimeout(() => {
            // Hide loading overlay
            if (overlay) overlay.style.display = 'none';
            
            // Navigate to predict section
            this.navigateToSection('predict');
            
            // Show demo notification
            this.showDemoNotification();
        }, 2000);
    }

    // Show demo notification
    showDemoNotification() {
        const notification = document.createElement('div');
        notification.className = 'demo-notification';
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-rocket"></i>
                <span>🚀 Live Demo Ready! Upload your sample_data.csv to see AI predict exoplanets!</span>
                <button onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 8 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 8000);
    }

    animateMetrics(metrics) {
        // Animate metric bars
        const metricBars = document.querySelectorAll('.metric-fill');
        const metricValues = document.querySelectorAll('.metric-value');
        
        metricBars.forEach((bar, index) => {
            const values = Object.values(metrics);
            if (values[index]) {
                setTimeout(() => {
                    bar.style.width = values[index] + '%';
                }, index * 200);
            }
        });

        // Animate metric values
        metricValues.forEach((value, index) => {
            const values = Object.values(metrics);
            if (values[index]) {
                this.animateNumber(value, 0, values[index], 1000, index * 200);
            }
        });
    }

    animateNumber(element, start, end, duration, delay = 0) {
        setTimeout(() => {
            const startTime = performance.now();
            const updateNumber = (currentTime) => {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const current = start + (end - start) * progress;
                
                element.textContent = current.toFixed(2) + '%';
                
                if (progress < 1) {
                    requestAnimationFrame(updateNumber);
                }
            };
            requestAnimationFrame(updateNumber);
        }, delay);
    }

    // Section-specific animations
    triggerSectionAnimations(sectionId) {
        switch (sectionId) {
            case 'analytics':
                this.animateAnalyticsCards();
                break;
            case 'predict':
                this.setupUploadHandlers();
                break;
            case 'about':
                this.animateAboutStats();
                break;
        }
    }

    animateAnalyticsCards() {
        const cards = document.querySelectorAll('.analytics-card');
        let animationIndex = 0;
        
        cards.forEach((card, index) => {
            // Skip animation for training info card completely
            if (card.classList.contains('training-info-card') || card.classList.contains('static-config')) {
                card.style.display = 'block';
                card.style.opacity = '1';
                card.style.visibility = 'visible';
                card.style.animation = 'none';
                card.style.transition = 'none';
                card.style.position = 'relative';
                card.style.zIndex = '10';
                return;
            }
            
            setTimeout(() => {
                card.classList.add('animate-slideInUp');
            }, animationIndex * 200);
            animationIndex++;
        });
        
        // Initialize charts when analytics section is shown
        setTimeout(() => {
            if (window.chartManager) {
                window.chartManager.animateChartsOnScroll();
            }
            // Also load training logs when analytics section is shown
            if (window.enhancedFeatures) {
                window.enhancedFeatures.loadTrainingLogs();
            }
        }, 500);
    }

    animateAboutStats() {
        const statItems = document.querySelectorAll('.stat-item');
        statItems.forEach((item, index) => {
            setTimeout(() => {
                item.classList.add('animate-slideInLeft');
            }, index * 150);
        });
    }

    // Upload handlers (will be implemented in upload.js)
    setupUploadHandlers() {
        // This will be handled by the upload.js file
        if (window.uploadHandler) {
            window.uploadHandler.init();
        }
    }

    // API Communication
    async makeApiRequest(endpoint, method = 'GET', data = null) {
        try {
            const options = {
                method,
                headers: {
                    'Content-Type': 'application/json',
                },
            };

            if (data) {
                options.body = JSON.stringify(data);
            }

            const response = await fetch(`${this.apiBaseUrl}${endpoint}`, options);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    // Utility Methods
    showLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.classList.add('active');
        }
    }

    hideLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.classList.remove('active');
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;

        // Add to page
        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);

        // Remove after delay
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.exoPlanetApp = new ExoPlanetApp();
});

// Global function for demo button
function runLiveDemo() {
    if (window.exoPlanetApp) {
        window.exoPlanetApp.runLiveDemo();
    }
}

// Global navigation function for buttons
function navigateToSection(sectionId) {
    if (window.exoPlanetApp) {
        window.exoPlanetApp.navigateToSection(sectionId);
    }
}

// Add notification styles
const notificationStyles = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--secondary-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: var(--spacing-md);
        z-index: 10000;
        transform: translateX(100%);
        transition: transform 0.3s ease-in-out;
        box-shadow: var(--shadow-lg);
    }

    .notification.show {
        transform: translateX(0);
    }

    .notification-content {
        display: flex;
        align-items: center;
        gap: var(--spacing-sm);
        color: var(--text-primary);
    }

    .notification-success {
        border-color: var(--success-color);
        background: rgba(107, 207, 127, 0.1);
    }

    .notification-error {
        border-color: var(--secondary-color);
        background: rgba(255, 107, 107, 0.1);
    }

    .notification-warning {
        border-color: var(--warning-color);
        background: rgba(255, 217, 61, 0.1);
    }

    .notification-info {
        border-color: var(--primary-color);
        background: rgba(0, 212, 255, 0.1);
    }
`;

// Inject notification styles
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);
