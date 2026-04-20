document.addEventListener('DOMContentLoaded', () => {
    // 复用 identity.js 中的核心元素和变量
    const dialogTemplate = document.getElementById('dialogTemplate');
    const contentArea = document.getElementById('contentArea');
    let highestZIndex = 101;

    // --- 复用的函数 (可以从 identity.js 复制过来) ---

    function makeDraggable(element) {
        const titleBar = element.querySelector('.dialog-title-bar');
        if (!titleBar) return;
        let offsetX, offsetY, isDragging = false;

        titleBar.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('dialog-close-button')) return;
            isDragging = true;
            offsetX = e.clientX - element.getBoundingClientRect().left;
            offsetY = e.clientY - element.getBoundingClientRect().top;
            element.style.zIndex = ++highestZIndex;
            titleBar.style.cursor = 'grabbing';
            document.querySelectorAll('.dialog-box.active').forEach(el => el.classList.remove('active'));
            element.classList.add('active');
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const containerRect = contentArea.getBoundingClientRect();
            let newX = e.clientX - offsetX - containerRect.left;
            let newY = e.clientY - offsetY - containerRect.top;

            // 简单的边界检测
            const elemWidth = element.offsetWidth;
            const elemHeight = element.offsetHeight;
            newX = Math.max(0, Math.min(newX, containerRect.width - elemWidth));
            newY = Math.max(0, Math.min(newY, containerRect.height - elemHeight));

            element.style.left = `${newX}px`;
            element.style.top = `${newY}px`;
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                titleBar.style.cursor = 'grab';
            }
        });
    }

    // --- 文章窗口创建函数 ---

    function createArticleWindow(title, contentHtml) {
        const dialogClone = dialogTemplate.content.cloneNode(true);
        const dialogBox = dialogClone.querySelector('.dialog-box');

        // 应用文章窗口的特定样式
        dialogBox.classList.add('article-window');

        // 设置标题和内容
        dialogBox.querySelector('.dialog-title').textContent = title;
        const contentElement = dialogBox.querySelector('.dialog-content');
        contentElement.innerHTML = contentHtml;
        // 添加一个包裹 div 以应用文章样式
        contentElement.classList.add('article-content');

        // 移除或隐藏不需要的按钮区域
        const buttonsContainer = dialogBox.querySelector('.dialog-buttons');
        if (buttonsContainer) {
            buttonsContainer.style.display = 'none';
        }

        // 设置关闭按钮
        const closeButton = dialogBox.querySelector('.dialog-close-button');
        closeButton.addEventListener('click', () => dialogBox.remove());

        // 激活窗口并提升 z-index
        dialogBox.addEventListener('mousedown', () => {
            dialogBox.style.zIndex = ++highestZIndex;
            document.querySelectorAll('.dialog-box.active').forEach(el => el.classList.remove('active'));
            dialogBox.classList.add('active');
        }, true);

        // 将窗口居中显示
        contentArea.appendChild(dialogBox);
        const containerRect = contentArea.getBoundingClientRect();
        const dialogRect = dialogBox.getBoundingClientRect();

        let initialX = (containerRect.width - dialogRect.width) / 2;
        let initialY = (containerRect.height - dialogRect.height) / 2;

        // 确保不会超出边界
        initialX = Math.max(10, initialX);
        initialY = Math.max(10, initialY);

        dialogBox.style.left = `${initialX}px`;
        dialogBox.style.top = `${initialY}px`;

        // 使窗口可拖动
        makeDraggable(dialogBox);

        return dialogBox;
    }

    // --- 页面加载时执行的逻辑 ---

    // 1. 激活侧边栏动画 (复用 identity.js 的逻辑)
    const navItems = document.querySelectorAll('.sidebar .nav-item');
    navItems.forEach((item, index) => {
        item.style.transitionDelay = `${index * 0.07}s`;
        item.classList.add('nav-item-visible');
    });

    // 2. 更新时钟 (复用 identity.js 的逻辑)
    // ... 这里可以复制 identity.js 中的时钟更新代码 ...

    // 3. 从模板中获取文章内容并创建窗口
    const articleTemplate = document.getElementById('articleContentTemplate');
    if (articleTemplate) {
        const articleContent = articleTemplate.innerHTML;
        // 延迟一点点弹出，效果更好
        setTimeout(() => {
            createArticleWindow("Blog.txt - 记事本", articleContent);
        }, 500); // 500毫秒后弹出
    } else {
        console.error("Article content template not found!");
    }
});