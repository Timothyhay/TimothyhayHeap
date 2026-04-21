document.addEventListener('DOMContentLoaded', () => {
    const listWindow = document.querySelector('.article-list-window');
    const contentArea = document.getElementById('contentArea');
    const previewBox = document.getElementById('hover-preview');
    const previewBody = previewBox.querySelector('.preview-body');
    const searchInput = document.getElementById('article-search');
    const viewModeToggle = document.getElementById('view-mode-toggle');
    const articleTable = document.getElementById('article-table');
    const articleTiles = document.getElementById('article-tiles');
    const statusFilter = document.getElementById('status-filter');
    const statusTotal = document.getElementById('status-total');
    const sidebarItems = document.querySelectorAll('.sidebar-item');
    
    let highestZIndex = 1000;
    let currentTag = 'ALL';
    
    // --- Window Dragging ---
    function makeDraggable(element) {
        const titleBar = element.querySelector('.dialog-title-bar');
        if (!titleBar) return;
        let offsetX, offsetY, isDragging = false;

        titleBar.addEventListener('mousedown', (e) => {
            if (element.classList.contains('maximized') || e.target.closest('button')) return;
            
            isDragging = true;
            const rect = element.getBoundingClientRect();
            offsetX = e.clientX - rect.left;
            offsetY = e.clientY - rect.top;
            element.style.zIndex = ++highestZIndex;
            titleBar.style.cursor = 'grabbing';
            element.style.transition = 'none';
            element.classList.add('active');
            previewBox.style.display = 'none';
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const areaRect = contentArea.getBoundingClientRect();
            let newX = (e.clientX - offsetX) - areaRect.left;
            let newY = (e.clientY - offsetY) - areaRect.top;
            newY = Math.max(-80, newY); 
            element.style.left = `${newX}px`;
            element.style.top = `${newY}px`;
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                titleBar.style.cursor = 'grab';
                element.style.transition = 'opacity 0.3s ease-out, transform 0.3s ease-out, width 0.3s ease-out, height 0.3s ease-out, top 0.3s ease-out, left 0.3s ease-out';
            }
        });
    }

    // --- Maximize Logic ---
    function toggleMaximize(windowElement) {
        const titleBar = windowElement.querySelector('.dialog-title-bar');
        const maximizeButton = windowElement.querySelector('.dialog-maximize-button');
        const isMaximized = windowElement.classList.contains('maximized');

        if (isMaximized) {
            windowElement.style.width = windowElement.dataset.originalWidth;
            windowElement.style.height = windowElement.dataset.originalHeight;
            windowElement.style.top = windowElement.dataset.originalTop;
            windowElement.style.left = windowElement.dataset.originalLeft;
            windowElement.classList.remove('maximized');
            maximizeButton.textContent = '▢';
            titleBar.style.cursor = 'grab';
        } else {
            windowElement.dataset.originalWidth = windowElement.style.width || `${windowElement.offsetWidth}px`;
            windowElement.dataset.originalHeight = windowElement.style.height || `${windowElement.offsetHeight}px`;
            windowElement.dataset.originalTop = windowElement.style.top;
            windowElement.dataset.originalLeft = windowElement.style.left;
            windowElement.style.width = `100%`;
            windowElement.style.height = `100%`;
            windowElement.style.top = `0px`;
            windowElement.style.left = `0px`;
            windowElement.classList.add('maximized');
            maximizeButton.textContent = '❐';
            titleBar.style.cursor = 'default';
        }
    }

    // --- Search and Filter Logic ---
    window.searchArticles = function() {
        applyFilters();
    };

    window.filterByTag = function(tag) {
        currentTag = tag;
        // Update sidebar UI
        sidebarItems.forEach(item => {
            if (item.textContent.trim().includes(tag) || (tag === 'ALL' && item.textContent.trim().includes('All Articles'))) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
        statusFilter.textContent = `Filter: ${tag}`;
        applyFilters();
    };

    function applyFilters() {
        const query = searchInput.value.toUpperCase();
        const rows = document.querySelectorAll('.article-row');
        const cards = document.querySelectorAll('.article-card');
        let visibleCount = 0;

        function checkItem(tags, title) {
            const matchesTag = currentTag === 'ALL' || tags.includes(currentTag);
            const matchesQuery = title.includes(query);
            return matchesTag && matchesQuery;
        }

        rows.forEach(row => {
            const tags = row.getAttribute('data-tags').split(',');
            const title = row.getAttribute('data-title');
            if (checkItem(tags, title)) {
                row.style.display = '';
                visibleCount++;
            } else {
                row.style.display = 'none';
            }
        });

        cards.forEach(card => {
            const tags = card.getAttribute('data-tags').split(',');
            const title = card.getAttribute('data-title');
            if (checkItem(tags, title)) {
                card.style.display = 'flex';
            } else {
                card.style.display = 'none';
            }
        });

        statusTotal.textContent = `Showing: ${visibleCount} Article(s)`;
    }

    // --- View Mode Toggle ---
    if (viewModeToggle) {
        viewModeToggle.addEventListener('click', () => {
            const isTableVisible = articleTable.style.display !== 'none';
            if (isTableVisible) {
                articleTable.style.display = 'none';
                articleTiles.style.display = 'grid';
                viewModeToggle.innerHTML = '<span class="btn-icon">☰</span> List';
            } else {
                articleTable.style.display = 'table';
                articleTiles.style.display = 'none';
                viewModeToggle.innerHTML = '<span class="btn-icon">▦</span> Tiles';
            }
        });
    }

    // --- Hover Preview ---
    function initPreview() {
        const rows = document.querySelectorAll('.article-row');
        rows.forEach(row => {
            row.addEventListener('mouseenter', (e) => {
                const text = row.getAttribute('data-preview');
                if (!text) return;
                previewBody.textContent = text + "...";
                previewBox.style.display = 'block';
                updatePreviewPosition(e);
            });
            row.addEventListener('mousemove', (e) => {
                updatePreviewPosition(e);
            });
            row.addEventListener('mouseleave', () => {
                previewBox.style.display = 'none';
            });
        });
    }

    function updatePreviewPosition(e) {
        const winRect = listWindow.getBoundingClientRect();
        let x = e.clientX - winRect.left + 20;
        let y = e.clientY - winRect.top + 20;
        if (x + 260 > winRect.width) x = e.clientX - winRect.left - 260;
        if (y + 150 > winRect.height) y = e.clientY - winRect.top - 150;
        previewBox.style.left = `${x}px`;
        previewBox.style.top = `${y}px`;
    }

    // --- Initialization ---
    if (listWindow) {
        makeDraggable(listWindow);
        const maxBtn = listWindow.querySelector('.dialog-maximize-button');
        if (maxBtn) maxBtn.addEventListener('click', () => toggleMaximize(listWindow));
        
        const closeBtn = listWindow.querySelector('.dialog-close-button');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                listWindow.style.opacity = '0';
                listWindow.style.transform = 'scale(0.95)';
                setTimeout(() => window.location.href = '/', 300);
            });
        }

        initPreview();
        
        const areaRect = contentArea.getBoundingClientRect();
        setTimeout(() => {
            const winRect = listWindow.getBoundingClientRect();
            listWindow.style.left = `${(areaRect.width - winRect.width) / 2}px`;
            listWindow.style.top = `40px`; 
            listWindow.style.opacity = '1';
            listWindow.style.transform = 'scale(1)';
        }, 100);
    }
});
