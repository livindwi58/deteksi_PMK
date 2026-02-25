// Main JavaScript file for the application

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded');
    
    // Initialize tooltips
    initTooltips();
    
    // Initialize popovers
    initPopovers();
    
    // Add smooth scrolling
    initSmoothScroll();
    
    // Auto-hide alerts after 5 seconds
    initAutoHideAlerts();
    
    // Add active class to current nav item
    setActiveNavItem();
    
    // Initialize file upload preview
    initFileUpload();
    
    // Initialize form validation
    initFormValidation();
});

// Initialize Bootstrap tooltips
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Initialize Bootstrap popovers
function initPopovers() {
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

// Smooth scrolling for anchor links
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Auto-hide alerts after 5 seconds
function initAutoHideAlerts() {
    // Only auto-hide top-of-page flash alerts (container.mt-3). Do NOT auto-hide
    // the diagnosis result container (#hasil) so users can read results at their pace.
    setTimeout(function() {
        document.querySelectorAll('.container.mt-3 .alert').forEach(alert => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 15000); // 15s so users have more time to read messages
}

// Set active class to current nav item
function setActiveNavItem() {
    const currentPath = window.location.pathname;
    document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
}

// Initialize file upload preview
function initFileUpload() {
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('preview');
                    if (preview) {
                        preview.src = e.target.result;
                        preview.classList.remove('d-none');
                    }
                    
                    // Show file info
                    const fileInfo = document.getElementById('fileInfo');
                    const fileName = document.getElementById('fileName');
                    if (fileInfo && fileName) {
                        fileName.textContent = this.files[0].name;
                        fileInfo.classList.remove('d-none');
                    }
                    
                    // Get image dimensions
                    const img = new Image();
                    img.onload = function() {
                        const dimensions = document.getElementById('imageDimensions');
                        if (dimensions) {
                            dimensions.textContent = `${this.width} x ${this.height} pixels`;
                        }
                    };
                    img.src = e.target.result;
                }.bind(this);
                reader.readAsDataURL(this.files[0]);
            }
        });
    }
}

// Initialize form validation
function initFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
}

// AJAX function for API calls
async function fetchAPI(url, method = 'GET', data = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json',
        }
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(url, options);
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('API Error:', error);
        showNotification('error', 'Terjadi kesalahan saat menghubungi server');
        return null;
    }
}

// Show notification
function showNotification(type, message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed top-0 end-0 m-3`;
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Format date
function formatDate(date) {
    return new Date(date).toLocaleString('id-ID', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('success', 'Teks berhasil disalin');
    }).catch(() => {
        showNotification('error', 'Gagal menyalin teks');
    });
}

// Download file
function downloadFile(content, fileName, contentType) {
    const a = document.createElement('a');
    const file = new Blob([content], { type: contentType });
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
    URL.revokeObjectURL(a.href);
}

// Debounce function for search inputs
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function for scroll events
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Add to favorites
function addToFavorites(title, url) {
    if (window.sidebar && window.sidebar.addPanel) { // Firefox
        window.sidebar.addPanel(title, url, '');
    } else if (window.external && ('AddFavorite' in window.external)) { // IE
        window.external.AddFavorite(url, title);
    } else { // Chrome, Safari
        showNotification('info', 'Tekan Ctrl+D untuk menambahkan ke bookmark');
    }
}

// Detect browser
function getBrowser() {
    const ua = navigator.userAgent;
    let browser = 'Unknown';
    
    if (ua.indexOf('Firefox') > -1) browser = 'Firefox';
    else if (ua.indexOf('Chrome') > -1) browser = 'Chrome';
    else if (ua.indexOf('Safari') > -1) browser = 'Safari';
    else if (ua.indexOf('MSIE') > -1 || ua.indexOf('Trident') > -1) browser = 'Internet Explorer';
    else if (ua.indexOf('Edge') > -1) browser = 'Edge';
    
    return browser;
}

// Check online status
window.addEventListener('online', () => {
    showNotification('success', 'Koneksi internet tersambung kembali');
});

window.addEventListener('offline', () => {
    showNotification('warning', 'Koneksi internet terputus');
});

// Lazy loading images
if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.add('fade-in');
                imageObserver.unobserve(img);
            }
        });
    });
    
    document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
    });
}

// Prevent double form submission
document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', function() {
        const submitButton = this.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = true;
            setTimeout(() => {
                submitButton.disabled = false;
            }, 3000);
        }
    });
});

// Dark mode toggle
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDark);
}

// Load dark mode preference
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
}

// Export functions for use in HTML
window.showNotification = showNotification;
window.formatDate = formatDate;
window.formatFileSize = formatFileSize;
window.copyToClipboard = copyToClipboard;
window.downloadFile = downloadFile;