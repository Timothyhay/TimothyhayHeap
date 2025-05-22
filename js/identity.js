document.addEventListener('DOMContentLoaded', () => {
    const contentArea = document.getElementById('contentArea');
    const dialogTemplate = document.getElementById('dialogTemplate');
    let highestZIndex = 100; // Start higher due to main logo

    // --- MODIFIED: Initialize highestZIndex for dialogs considering logo ---
    // The main logo is at z-index 100. Dialogs should start above that when active.
    // The dialog's own base z-index is set in CSS (e.g., 20), then dynamically increased.
    // Let's make sure highestZIndex for dynamic assignment starts above the logo.
    // Or, ensure active dialogs get a very high z-index.
    // For now, dialogs will start with CSS z-index, and `highestZIndex` will increment from there.


    const MAX_DIALOGS_TO_SPAWN = 6;
    let spawnedDialogsCount = 0;
    const dialogsData = [ /* ... your dialogsData ... */
        { id: 1, title: '💖 宇宙卵团子巡演', width: '340px', content: `<img src="https://via.placeholder.com/300x180/b4f8c8/333333?text=宇宙卵团子" alt="Cosmic Dorodango"><p>ZUTOMAYO FACTORY「宇宙の卵子 DORODANGO」巡演开始！更多信息请查看官网。</p>` },
        { id: 2, title: '📅 3.29-5.18 竞技场之旅', width: '360px', content: `<img src="https://via.placeholder.com/320x150/a0e7e5/333333?text=竞技场之旅" alt="Arena Tour"><p>ZUTOMAYO 竞技场之旅 2024「本格的に」即将举行。日程: 3月29日 - 5月18日</p>` },
        { id: 3, title: '📢 News 2025.05.18', width: '300px', content: `<p><strong>"YAKI YAK" 师父和师父</strong></p><p>新曲发布！详情请关注后续公告。</p>` },
        { id: 4, title: '💿 1st ZUTOMAYO', width: '320px', content: `<img src="https://via.placeholder.com/280x200/d7b0ff/333333?text=潜潜話专辑" alt="Album Art"><p>首张专辑「潜潜話」好评发售中！探索ZUTOMAYO的音乐世界。</p>` },
        { id: 5, title: 'MV发布 5/22 21:00', width: '350px', content: `<p><strong>你能和 Cream 一起来看我吗？</strong></p><p>新MV将于 5月22日 21:00 (JST) 发布！敬请期待！不要错过！</p>` },
        { id: 6, title: '✨ 特别通知 ✨', width: '310px', content: `<p>感谢大家一直以来的支持！</p><p>未来将有更多精彩内容，请保持关注官方动态！</p>` }
    ];
    let availableDialogIndices = [];

    function initializeAvailableDialogs() {
        availableDialogIndices = dialogsData.map((_, index) => index);
        for (let i = availableDialogIndices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [availableDialogIndices[i], availableDialogIndices[j]] = [availableDialogIndices[j], availableDialogIndices[i]];
        }
    }
    initializeAvailableDialogs();

    let dialogCreationInterval;

    // --- NEW: Animate Sidebar Items ---
    function animateSidebarItems() {
        const navItems = document.querySelectorAll('.sidebar .nav-item');
        navItems.forEach((item, index) => {
            // Apply a staggered delay for each item
            item.style.transitionDelay = `${index * 0.07}s`; // 70ms delay between items
            item.classList.add('nav-item-visible');
        });
    }
    // Call the animation function after a short delay from DOMContentLoaded
    // to ensure styles are applied and allow main logo animation to start.
    setTimeout(animateSidebarItems, 700); // Start sidebar animation after 0.7s


    function createDialog(data) {
        const dialogClone = dialogTemplate.content.cloneNode(true);
        const dialogBox = dialogClone.querySelector('.dialog-box');
        const titleElement = dialogBox.querySelector('.dialog-title');
        const contentElement = dialogBox.querySelector('.dialog-content');
        const closeButton = dialogBox.querySelector('.dialog-close-button');

        titleElement.textContent = data.title || '无标题对话框';
        contentElement.innerHTML = data.content || '<p>无内容。</p>';
        if (data.width) {
            dialogBox.style.width = data.width;
        }

        // Ensure dialog width is parsed correctly for positioning
        const dialogComputedStyle = window.getComputedStyle(dialogBox);
        // Temporarily append to measure, then remove if not yet added to DOM
        // This is a bit complex; simpler to use default or data.width for initial calc
        const dialogWidth = parseInt(data.width) || parseInt(dialogComputedStyle.width) || 300;
        const dialogHeight = 180; // Estimate, actual height depends on content

        const maxX = contentArea.offsetWidth - dialogWidth - 20; // Add some margin
        const maxY = contentArea.offsetHeight - dialogHeight - 20;
        dialogBox.style.left = `${Math.max(5, Math.random() * Math.max(5, maxX))}px`;
        dialogBox.style.top = `${Math.max(5, Math.random() * Math.max(5, maxY))}px`;

        // MODIFIED: Ensure dialogs are above the main logo when active
        // The highestZIndex for dialogs should start above the main logo's z-index (100)
        // if a dialog is clicked. Base z-index is in CSS.
        let currentDialogBaseZIndex = parseInt(window.getComputedStyle(dialogBox).zIndex) || 20;
        if (highestZIndex <= 100) highestZIndex = 101; // Ensure it starts above logo
        dialogBox.style.zIndex = currentDialogBaseZIndex; // Initial from CSS

        makeDraggable(dialogBox);

        closeButton.addEventListener('click', () => {
            dialogBox.remove();
        });

        dialogBox.addEventListener('mousedown', () => {
            // When a dialog is clicked, bring it to the very front
            dialogBox.style.zIndex = ++highestZIndex;
            dialogBox.classList.add('active');
        }, true);

        dialogBox.addEventListener('mouseup', () => {
            dialogBox.classList.remove('active');
        });

        contentArea.appendChild(dialogBox);
        return dialogBox;
    }

    function makeDraggable(element) {
        // ... (makeDraggable function remains largely the same, ensure it handles boundaries well)
        const titleBar = element.querySelector('.dialog-title-bar');
        let offsetX, offsetY, isDragging = false;

        titleBar.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('dialog-close-button')) return;
            isDragging = true;
            offsetX = e.clientX - element.offsetLeft;
            offsetY = e.clientY - element.offsetTop;
            element.style.zIndex = ++highestZIndex;
            titleBar.style.cursor = 'grabbing';
            element.classList.add('active');
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            let newX = e.clientX - offsetX;
            let newY = e.clientY - offsetY;
            const parentRect = contentArea.getBoundingClientRect();
            const elemRect = element.getBoundingClientRect();

            newX = Math.max(0, Math.min(newX, parentRect.width - elemRect.width));
            newY = Math.max(0, Math.min(newY, parentRect.height - elemRect.height));

            element.style.left = `${newX}px`;
            element.style.top = `${newY}px`;
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                titleBar.style.cursor = 'grab';
                element.classList.remove('active');
            }
        });
    }

    function spawnNextDialog() {
        // MODIFIED: Stop if max count reached OR no more unique dialogs
        if (spawnedDialogsCount >= MAX_DIALOGS_TO_SPAWN || availableDialogIndices.length === 0) {
            console.log(`已弹出 ${spawnedDialogsCount} 个窗口。达到上限或无独特窗口可弹出，停止。`);
            clearInterval(dialogCreationInterval);
            return;
        }

        const nextDialogDataOriginalIndex = availableDialogIndices.shift(); // Get and remove first (shuffled) index
        const dialogDataToShow = dialogsData[nextDialogDataOriginalIndex];

        if (dialogDataToShow) {
            createDialog(dialogDataToShow);
            spawnedDialogsCount++;
            console.log(`已弹出 ${spawnedDialogsCount} / ${MAX_DIALOGS_TO_SPAWN} 个窗口 (ID: ${dialogDataToShow.id})。剩余独特窗口: ${availableDialogIndices.length}`);
        } else {
            // Should not happen if availableDialogIndices is managed correctly
            console.warn("尝试获取的对话框数据不存在。");
        }
    }

    // Initial dialog spawning and interval
    if (dialogsData.length > 0) {
        spawnNextDialog();
        setTimeout(() => {
            if (spawnedDialogsCount < MAX_DIALOGS_TO_SPAWN && availableDialogIndices.length > 0) {
                spawnNextDialog();
            }
        }, 800); // Faster second pop-up: 0.8 seconds

        dialogCreationInterval = setInterval(spawnNextDialog, 1800); // Faster interval: 1.8 seconds
    } else {
        console.log("没有可供弹出的对话框数据。");
    }

    // Window resize handling (remains the same)
    window.addEventListener('resize', () => {
        document.querySelectorAll('.dialog-box').forEach(dialog => {
            const dialogWidth = dialog.offsetWidth;
            const dialogHeight = dialog.offsetHeight;
            const maxX = contentArea.offsetWidth - dialogWidth - 10;
            const maxY = contentArea.offsetHeight - dialogHeight - 10;
            let currentX = parseInt(dialog.style.left) || 0;
            let currentY = parseInt(dialog.style.top) || 0;
            dialog.style.left = `${Math.min(Math.max(0, currentX), Math.max(0, maxX))}px`;
            dialog.style.top = `${Math.min(Math.max(0, currentY), Math.max(0, maxY))}px`;
        });
    });
});