document.addEventListener('DOMContentLoaded', () => {
    const listWindow = document.querySelector('.article-list-window');
    const contentArea = document.getElementById('contentArea');
    const mainLogo = document.querySelector('.main-logo');
    const previewBox = document.getElementById('hover-preview');
    const previewBody = previewBox.querySelector('.preview-body');
    const sidebar = document.querySelector('.sidebar');
    const copyrightEl = document.querySelector('.copyright');
    let highestZIndex = 1000;
    
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
            document.querySelectorAll('.dialog-box.active').forEach(el => el.classList.remove('active'));
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

    // --- Unified Maximize Logic ---
    function toggleMaximize(windowElement) {
        const titleBar = windowElement.querySelector('.dialog-title-bar');
        const maximizeButton = windowElement.querySelector('.dialog-maximize-button');
        const isMaximized = windowElement.classList.contains('maximized');

        if (isMaximized) {
            // Restore
            windowElement.style.width = windowElement.dataset.originalWidth;
            windowElement.style.height = windowElement.dataset.originalHeight;
            windowElement.style.top = windowElement.dataset.originalTop;
            windowElement.style.left = windowElement.dataset.originalLeft;

            windowElement.classList.remove('maximized');
            document.body.classList.remove('window-maximized');
            
            maximizeButton.textContent = '▢';
            titleBar.style.cursor = 'grab';
            
            if (copyrightEl && copyrightEl.dataset.originalHtml) {
                copyrightEl.innerHTML = copyrightEl.dataset.originalHtml;
            }
        } else {
            // Maximize
            windowElement.dataset.originalWidth = windowElement.style.width || `${windowElement.offsetWidth}px`;
            windowElement.dataset.originalHeight = windowElement.style.height || `${windowElement.offsetHeight}px`;
            windowElement.dataset.originalTop = windowElement.style.top;
            windowElement.dataset.originalLeft = windowElement.style.left;

            // 因为窗口在 #contentArea 内部，且 #contentArea 已经是 flex 子元素
            // 所以 left: 0 和 top: 0 就是侧边栏右侧的最顶端
            windowElement.style.width = `100%`;
            windowElement.style.height = `100%`;
            windowElement.style.top = `0px`;
            windowElement.style.left = `0px`;

            windowElement.style.transition = 'all 0.4s ease-in-out';
            windowElement.classList.add('maximized');
            document.body.classList.add('window-maximized');

            maximizeButton.textContent = '❐';
            titleBar.style.cursor = 'default';

            if (copyrightEl) {
                if (!copyrightEl.dataset.originalHtml) {
                    copyrightEl.dataset.originalHtml = copyrightEl.innerHTML;
                }
                copyrightEl.innerHTML = '© 2025 TangerineSoda. All Rights Reserved';
            }
        }
    }

    // --- 悬浮预览逻辑 ---
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
        if (x + 240 > winRect.width) x = e.clientX - winRect.left - 240;
        if (y + 150 > winRect.height) y = e.clientY - winRect.top - 150;
        previewBox.style.left = `${x}px`;
        previewBox.style.top = `${y}px`;
    }

    if (listWindow) {
        makeDraggable(listWindow);
        const maxBtn = listWindow.querySelector('.dialog-maximize-button');
        if (maxBtn) {
            maxBtn.addEventListener('click', () => toggleMaximize(listWindow));
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

    const closeBtn = listWindow.querySelector('.dialog-close-button');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            listWindow.style.opacity = '0';
            listWindow.style.transform = 'scale(0.95)';
            document.body.classList.remove('window-maximized');
            setTimeout(() => {
                window.location.href = '/';
            }, 300);
        });
    }
});
