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