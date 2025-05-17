// Main JavaScript file for the application

// When the document is ready
document.addEventListener('DOMContentLoaded', function() {
    // Toggle mobile navigation
    const navbarToggler = document.querySelector('.navbar-toggler');
    if (navbarToggler) {
        navbarToggler.addEventListener('click', function() {
            document.querySelector('.navbar-collapse').classList.toggle('show');
        });
    }
    
    // Setup file input label update
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name || 'No file chosen';
            const label = e.target.nextElementSibling;
            if (label && label.classList.contains('form-file-label')) {
                label.textContent = fileName;
            }
        });
    });
    
    // Enable tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        tooltips.forEach(tooltip => {
            new bootstrap.Tooltip(tooltip);
        });
    }
    
    // Enable popovers if Bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Popover) {
        const popovers = document.querySelectorAll('[data-bs-toggle="popover"]');
        popovers.forEach(popover => {
            new bootstrap.Popover(popover);
        });
    }
    
    // Dark mode toggle
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    if (darkModeToggle) {
        // Check for saved user preference
        const savedDarkMode = localStorage.getItem('darkMode') === 'true';
        
        // Apply saved preference
        if (savedDarkMode) {
            document.body.classList.add('dark-mode');
            darkModeToggle.checked = true;
        }
        
        // Handle toggle changes
        darkModeToggle.addEventListener('change', function() {
            if (this.checked) {
                document.body.classList.add('dark-mode');
                localStorage.setItem('darkMode', 'true');
            } else {
                document.body.classList.remove('dark-mode');
                localStorage.setItem('darkMode', 'false');
            }
        });
    }

    // Add loading indicator for file uploads
    const uploadForm = document.querySelector('form[action="/upload"]');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function() {
            showLoading('Uploading and processing file...');
        });
    }
    
    // Add loading indicator for sample data loading
    const sampleLink = document.querySelector('a[href="/sample"]');
    if (sampleLink) {
        sampleLink.addEventListener('click', function() {
            showLoading('Loading sample data...');
        });
    }
    
    // Dashboard feature card animation
    const animateFeatureCards = () => {
        const featureCards = document.querySelectorAll('.feature-card');
        
        if (featureCards.length > 0) {
            featureCards.forEach((card, index) => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                
                setTimeout(() => {
                    card.style.transition = 'all 0.5s ease';
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, 100 + (index * 100));
            });
        }
    };
    
    // Animate dashboard stats counting
    const animateCounters = () => {
        const statNumbers = document.querySelectorAll('.stats-number');
        
        if (statNumbers.length > 0) {
            statNumbers.forEach(number => {
                const target = parseInt(number.textContent, 10);
                let count = 0;
                const duration = 1500; // ms
                const frameDuration = 1000 / 60; // 60fps
                const totalFrames = Math.round(duration / frameDuration);
                const increment = target / totalFrames;
                
                const counter = setInterval(() => {
                    count += increment;
                    
                    if (count >= target) {
                        number.textContent = target;
                        clearInterval(counter);
                    } else {
                        number.textContent = Math.floor(count);
                    }
                }, frameDuration);
            });
        }
    };
    
    // Run animations if we're on the dashboard
    if (window.location.pathname === '/' || window.location.pathname === '/index') {
        setTimeout(animateFeatureCards, 300);
        setTimeout(animateCounters, 500);
    }
    
    // Add AJAX error handler to display error messages
    $(document).ajaxError(function(event, jqXHR, settings, thrownError) {
        hideLoading();
        // Display error message
        const message = jqXHR.responseJSON && jqXHR.responseJSON.error 
            ? jqXHR.responseJSON.error 
            : 'An error occurred while processing your request.';
            
        // Create an alert to display the error
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show';
        alertDiv.setAttribute('role', 'alert');
        alertDiv.innerHTML = `
            <strong>Error:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Find a good place to show the alert - typically at the top of the content
        const content = document.querySelector('#content');
        if (content && content.firstChild) {
            content.insertBefore(alertDiv, content.firstChild);
        } else {
            // Fallback - add to body
            document.body.insertBefore(alertDiv, document.body.firstChild);
        }
    });
    
    // Handle AJAX requests with loading indicators
    $(document).ajaxSend(function(event, jqXHR, settings) {
        if (settings.url.includes('/api/data_preview')) {
            showLoading('Loading data preview...');
        } else if (settings.url.includes('/api/clean_data')) {
            showLoading('Cleaning data...');
        } else if (settings.url.includes('/api/convert_types')) {
            showLoading('Converting data types...');
        } else if (settings.url.includes('/api/ml/')) {
            showLoading('Processing machine learning request...');
        }
    });
    
    $(document).ajaxComplete(function(event, jqXHR, settings) {
        hideLoading();
    });

    // Add any initialization code here
    console.log('Inventory Optimization app initialized with enhanced dashboard');
});

// Utility function to format currency
function formatCurrency(amount, currency = 'USD') {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency
    }).format(amount);
}

// Utility function to format numbers with commas
function formatNumber(number, decimals = 0) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(number);
}

// Utility function to format percentages
function formatPercent(number, decimals = 2) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(number / 100);
}

// Utility function to create simple charts
function createChart(elementId, type, labels, datasets, options = {}) {
    if (!window.Chart) {
        console.error('Chart.js is not loaded');
        return null;
    }
    
    const ctx = document.getElementById(elementId).getContext('2d');
    return new Chart(ctx, {
        type: type,
        data: {
            labels: labels,
            datasets: datasets
        },
        options: options
    });
}

// Utility function to download data as CSV
function downloadCSV(data, filename = 'data.csv') {
    let csvContent = "data:text/csv;charset=utf-8,";
    
    // Handle array of objects
    if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'object') {
        // Get headers
        const headers = Object.keys(data[0]);
        csvContent += headers.join(',') + '\r\n';
        
        // Add rows
        data.forEach(item => {
            const row = headers.map(header => {
                // Handle values with commas by wrapping in quotes
                const cell = item[header] !== undefined ? item[header] : '';
                return typeof cell === 'string' && cell.includes(',') 
                    ? `"${cell}"` 
                    : cell;
            });
            csvContent += row.join(',') + '\r\n';
        });
    } 
    // Handle 2D array
    else if (Array.isArray(data)) {
        data.forEach(row => {
            const formattedRow = row.map(cell => {
                return typeof cell === 'string' && cell.includes(',') 
                    ? `"${cell}"` 
                    : cell;
            });
            csvContent += formattedRow.join(',') + '\r\n';
        });
    }
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Show loading indicator
function showLoading(message = "Loading data...") {
    // Create loading overlay if it doesn't exist
    if (!document.getElementById('loading-overlay')) {
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.style.position = 'fixed';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100%';
        overlay.style.height = '100%';
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        overlay.style.display = 'flex';
        overlay.style.justifyContent = 'center';
        overlay.style.alignItems = 'center';
        overlay.style.zIndex = '9999';
        
        const spinner = document.createElement('div');
        spinner.className = 'spinner-border text-light';
        spinner.setAttribute('role', 'status');
        
        const loadingText = document.createElement('div');
        loadingText.id = 'loading-text';
        loadingText.className = 'ms-3 text-light';
        loadingText.textContent = message;
        
        const content = document.createElement('div');
        content.className = 'd-flex align-items-center';
        content.appendChild(spinner);
        content.appendChild(loadingText);
        
        overlay.appendChild(content);
        document.body.appendChild(overlay);
    } else {
        document.getElementById('loading-text').textContent = message;
        document.getElementById('loading-overlay').style.display = 'flex';
    }
}

// Hide loading indicator
function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// Dashboard-specific functions
function updateDashboardStats(stats) {
    // Update dashboard stats if they exist
    Object.keys(stats).forEach(key => {
        const element = document.getElementById(`stat-${key}`);
        if (element) {
            element.textContent = stats[key];
        }
    });
}

// Function to animate progress bars
function animateProgressBars() {
    const progressBars = document.querySelectorAll('.progress-bar');
    
    progressBars.forEach(bar => {
        const width = bar.getAttribute('data-width') || bar.style.width;
        bar.style.width = '0%';
        
        setTimeout(() => {
            bar.style.width = width;
        }, 300);
    });
}

$(document).ready(function() {
    // Sidebar toggle
    $('#sidebarCollapse').on('click', function() {
        $('#sidebar').toggleClass('active');
        $('#content').toggleClass('active');
        $(this).toggleClass('active');
        
        // Update toggle button text based on sidebar state
        if ($('#sidebar').hasClass('active')) {
            $(this).find('span').text('Show Sidebar');
        } else {
            $(this).find('span').text('Hide Sidebar');
        }
    });

    // Handle dropdown toggles
    $('.dropdown-toggle').on('click', function(e) {
        e.preventDefault();
        $(this).parent().find('.collapse').collapse('toggle');
    });

    // Close other dropdowns when opening a new one
    $('.collapse').on('show.bs.collapse', function() {
        $('.collapse.show').not(this).collapse('hide');
    });

    // Add active class to current page
    var currentLocation = window.location.pathname;
    $('.nav-link').each(function() {
        var link = $(this).attr('href');
        if (currentLocation === link) {
            $(this).addClass('active');
            // If this is inside a dropdown, expand the dropdown
            var dropdown = $(this).closest('.collapse');
            if (dropdown.length) {
                dropdown.addClass('show');
                dropdown.prev('.dropdown-toggle').addClass('active');
            }
        }
    });
    
    // On small screens, start with sidebar collapsed
    if ($(window).width() < 768) {
        $('#sidebar').addClass('active');
        $('#content').addClass('active');
        $('#sidebarCollapse').addClass('active');
        $('#sidebarCollapse').find('span').text('Show Sidebar');
    }
    
    // Animate progress bars when on dashboard
    if (window.location.pathname === '/' || window.location.pathname === '/index') {
        setTimeout(animateProgressBars, 800);
    }
    
    // Hover effects for feature cards
    $('.feature-card').hover(
        function() {
            $(this).find('i').addClass('fa-bounce');
        },
        function() {
            $(this).find('i').removeClass('fa-bounce');
        }
    );
}); 