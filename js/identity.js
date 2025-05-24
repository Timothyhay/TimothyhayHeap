// Wait for the HTML document to be fully loaded and parsed before executing the script
document.addEventListener('DOMContentLoaded', () => {
    // Get references to key DOM elements
    const contentArea = document.getElementById('contentArea'); // Area where dialogs will appear
    const dialogTemplate = document.getElementById('dialogTemplate'); // HTML template for dialogs

    // Initialize z-index counter for dialog stacking.
    // Starts above the main logo's z-index (100) to ensure dialogs can come to the front.
    let highestZIndex = 101;

    // Configuration for dialog spawning
    const MAX_DIALOGS_TO_SPAWN = 6; // Maximum number of dialogs that will be spawned in total
    let spawnedDialogsCount = 0;   // Counter for how many dialogs have been spawned

    // Data for the dialogs. Each object represents one dialog.
    // 'id' is for potential future reference, 'title', 'width', and 'content' (HTML string) define the dialog.
    const dialogsData = [
        { id: 1, title: '💖 宇宙卵团子巡演', width: '340px', content: `<img src="https://via.placeholder.com/300x180/b4f8c8/333333?text=宇宙卵团子巡演图" alt="宇宙卵团子巡演"><p>ZUTOMAYO FACTORY「宇宙の卵子 DORODANGO」巡演开始！更多信息请查看官网。</p>` },
        { id: 2, title: '📅 3.29-5.18 竞技场之旅', width: '360px', content: `<img src="https://via.placeholder.com/320x150/a0e7e5/333333?text=竞技场之旅图" alt="竞技场之旅"><p>ZUTOMAYO 竞技场之旅 2024「本格的に」即将举行。日程: 3月29日 - 5月18日</p>` },
        { id: 3, title: '📢 News 2025.05.18', width: '300px', content: `<p><strong>"YAKI YAK" 师父和师父</strong></p><p>新曲发布！详情请关注后续公告。</p>` },
        { id: 4, title: '💿 1st ZUTOMAYO', width: '320px', content: `<img src="https://via.placeholder.com/280x200/d7b0ff/333333?text=潜潜話专辑图" alt="潜潜話专辑"><p>首张专辑「潜潜話」好评发售中！探索ZUTOMAYO的音乐世界。</p>` },
        { id: 5, title: 'MV发布 5/22 21:00', width: '350px', content: `<p><strong>你能和 Cream 一起来看我吗？</strong></p><p>新MV将于 5月22日 21:00 (JST) 发布！敬请期待！不要错过！</p>` },
        { id: 6, title: '✨ 特别通知 ✨', width: '310px', content: `<p>感谢大家一直以来的支持！</p><p>未来将有更多精彩内容，请保持关注官方动态！</p>` },
        { id: 7, title: '🎶 新歌试听片段', width: '330px', content: `<p>最新单曲片段抢先听！</p><p>感受ZUTOMAYO的独特魅力。</p><img src="https://via.placeholder.com/290x100/f9c5d1/333333?text=新歌试听图" alt="新歌试听">` }
    ];

    // Array to store indices of dialogs from dialogsData that are still available to be shown.
    // This helps ensure dialogs don't repeat until all unique ones are shown (or MAX_DIALOGS_TO_SPAWN is hit).
    let availableDialogIndices = [];
    let dialogCreationInterval; // Interval timer for spawning dialogs

    // --- Function to initialize and shuffle available dialog indices ---
    function initializeAvailableDialogs() {
        availableDialogIndices = dialogsData.map((_, index) => index); // Create an array of indices [0, 1, 2, ...]
        // Fisher-Yates shuffle algorithm to randomize the order of dialogs
        for (let i = availableDialogIndices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1)); // Pick a random index before the current one
            [availableDialogIndices[i], availableDialogIndices[j]] = [availableDialogIndices[j], availableDialogIndices[i]]; // Swap elements
        }
    }
    initializeAvailableDialogs(); // Call on script load

    // --- Function to animate sidebar navigation items sequentially ---
    function animateSidebarItems() {
        const navItems = document.querySelectorAll('.sidebar .nav-item'); // Get all elements with .nav-item class in sidebar
        console.log(`[animateSidebarItems] Found ${navItems.length} nav items to animate.`); // Debug log

        navItems.forEach((item, index) => {
            // Log for debugging each item being processed
            console.log(`[animateSidebarItems] Animating item ${index}:`, item.textContent.trim().substring(0,20) + "...");
            // Set the individual transition-delay for each item using an inline style
            // This creates the staggered animation effect.
            item.style.transitionDelay = `${index * 0.07}s`; // e.g., 0s, 0.07s, 0.14s, ...
            // Add the 'nav-item-visible' class to trigger the CSS transition defined in style.css
            item.classList.add('nav-item-visible');
        });
    }
    // Start sidebar animation after a short delay to allow other elements to render and
    // potentially for the main logo (if it were animated) to start.
    setTimeout(animateSidebarItems, 700); // 0.7 seconds delay

    // --- Function to create and display a new dialog box ---
    function createDialog(data) {
        const dialogClone = dialogTemplate.content.cloneNode(true); // Clone the hidden dialog template
        const dialogBox = dialogClone.querySelector('.dialog-box');
        const titleElement = dialogBox.querySelector('.dialog-title');
        const contentElement = dialogBox.querySelector('.dialog-content');
        const closeButton = dialogBox.querySelector('.dialog-close-button');

        // Populate dialog with data
        titleElement.textContent = data.title || 'Untitled Dialog'; // Set title, or default
        contentElement.innerHTML = data.content || '<p>No content.</p>'; // Set content HTML, or default
        if (data.width) {
            dialogBox.style.width = data.width; // Set custom width if provided
        }

        // Calculate random initial position for the dialog within the content area
        const dialogWidth = parseInt(dialogBox.style.width) || 300; // Use specified or default width
        const dialogHeight = 180; // Approximate minimum height for positioning calculation
        const maxX = contentArea.offsetWidth - dialogWidth - 20;  // Max X, with some margin
        const maxY = contentArea.offsetHeight - dialogHeight - 20; // Max Y, with some margin
        dialogBox.style.left = `${Math.max(5, Math.random() * Math.max(5, maxX))}px`; // Random X, at least 5px from edge
        dialogBox.style.top = `${Math.max(5, Math.random() * Math.max(5, maxY))}px`;  // Random Y, at least 5px from edge

        // The base z-index for .dialog-box is set in CSS (e.g., 20).
        // When a dialog is interacted with, 'highestZIndex' will be used to bring it to the front.

        makeDraggable(dialogBox); // Make the dialog draggable

        // Event listener for the close button
        closeButton.addEventListener('click', () => {
            dialogBox.remove(); // Remove the dialog from the DOM
        });

        // Event listener to bring dialog to front when clicked (mousedown)
        dialogBox.addEventListener('mousedown', () => {
            dialogBox.style.zIndex = ++highestZIndex; // Increment and assign new highest z-index
            dialogBox.classList.add('active');      // Add 'active' class (for optional styling)
        }, true); // Use capture phase to ensure this fires before drag starts on title bar

        // Event listener to remove 'active' class on mouseup (optional)
        dialogBox.addEventListener('mouseup', () => {
            dialogBox.classList.remove('active');
        });

        contentArea.appendChild(dialogBox); // Add the new dialog to the content area
        return dialogBox;
    }

    // --- Function to make a dialog element draggable ---
    function makeDraggable(element) {
        const titleBar = element.querySelector('.dialog-title-bar');
        let offsetX, offsetY, isDragging = false;

        titleBar.addEventListener('mousedown', (e) => {
            // Prevent dragging if the click target is the close button itself
            if (e.target.classList.contains('dialog-close-button')) return;

            isDragging = true;
            // Calculate mouse offset relative to the dialog's top-left corner
            offsetX = e.clientX - element.offsetLeft;
            offsetY = e.clientY - element.offsetTop;

            // Bring the dragged dialog to the very front
            element.style.zIndex = ++highestZIndex;
            titleBar.style.cursor = 'grabbing'; // Change cursor to 'grabbing'
            element.classList.add('active');    // Add 'active' class
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return; // Only move if dragging is active

            let newX = e.clientX - offsetX;
            let newY = e.clientY - offsetY;

            // Constrain dialog movement within the bounds of the contentArea
            const parentRect = contentArea.getBoundingClientRect(); // Content area dimensions
            const elemRect = element.getBoundingClientRect();     // Dialog dimensions

            newX = Math.max(0, Math.min(newX, parentRect.width - elemRect.width)); // Clamp X
            newY = Math.max(0, Math.min(newY, parentRect.height - elemRect.height)); // Clamp Y

            element.style.left = `${newX}px`;
            element.style.top = `${newY}px`;
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                titleBar.style.cursor = 'grab'; // Reset cursor
                element.classList.remove('active'); // Remove 'active' class
            }
        });
    }

    // --- Function to spawn the next dialog from the available list ---
    function spawnNextDialog() {
        // Stop if max dialog count is reached OR no more unique dialogs are available
        if (spawnedDialogsCount >= MAX_DIALOGS_TO_SPAWN || availableDialogIndices.length === 0) {
            console.log(`Spawned ${spawnedDialogsCount} dialogs. Limit reached or no unique dialogs left. Stopping.`);
            clearInterval(dialogCreationInterval); // Stop the interval timer
            return;
        }

        // Get the index of the next dialog to show (from the shuffled list) and remove it
        const nextDialogDataOriginalIndex = availableDialogIndices.shift();
        const dialogDataToShow = dialogsData[nextDialogDataOriginalIndex];

        if (dialogDataToShow) {
            createDialog(dialogDataToShow); // Create and display the dialog
            spawnedDialogsCount++;          // Increment count of spawned dialogs
            console.log(`Spawned ${spawnedDialogsCount} / ${MAX_DIALOGS_TO_SPAWN} dialogs (ID: ${dialogDataToShow.id}). Unique dialogs remaining: ${availableDialogIndices.length}`);
        } else {
            // This case should ideally not be reached if logic is correct
            console.warn("Attempted to get dialog data that does not exist.");
        }
    }

    // --- Initial dialog spawning and interval setup ---
    if (dialogsData.length > 0) { // Only proceed if there's dialog data
        spawnNextDialog(); // Spawn the first dialog immediately

        // Spawn a second dialog after a short delay, if limits not reached
        setTimeout(() => {
            if (spawnedDialogsCount < MAX_DIALOGS_TO_SPAWN && availableDialogIndices.length > 0) {
                spawnNextDialog();
            }
        }, 1200); // 1.2 seconds delay for the second dialog

        // Set an interval to spawn subsequent dialogs
        dialogCreationInterval = setInterval(spawnNextDialog, 2200); // Spawn a new dialog every 2.2 seconds
    } else {
        console.log("No dialog data available to spawn.");
    }

    // --- NEW: Pixel Clock Functionality ---
    // --- NEW/UPDATED: Pixel Clock Functionality ---
    const yearElem = document.getElementById('clock-year');
    const weekElem = document.getElementById('clock-week');
    const hoursElem = document.getElementById('clock-hours');
    const minutesElem = document.getElementById('clock-minutes');
    const secondsElem = document.getElementById('clock-seconds');
    const cosmicRayElem = document.getElementById('cosmic-ray-intensity');

    // Function to get the ISO week number
    // Source: https://stackoverflow.com/a/6117889/1238098
    function getWeekNumber(d) {
        d = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
        d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7));
        var yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
        var weekNo = Math.ceil((((d - yearStart) / 86400000) + 1) / 7);
        return weekNo;
    }

    // Function to calculate fictional cosmic ray intensity
    function calculateCosmicRayIntensity(now) {
        // This is a completely arbitrary calculation for fun.
        // Uses minutes, seconds, and a bit of randomization.
        const minutes = now.getMinutes();
        const seconds = now.getSeconds();

        // Base value fluctuates with minutes
        let baseIntensity = (minutes % 10) * 0.15; // 0 to 1.35

        // Add fluctuation based on seconds (more chaotic)
        let secondFluctuation = Math.sin(seconds * (Math.PI / 15)) * 0.5; // -0.5 to 0.5, cycles every 30s

        // Add a very slow-changing component (e.g., based on hour)
        let hourComponent = (now.getHours() % 6) * 0.05; // 0 to 0.25, changes every 6 hours

        // Combine them and add some randomness
        let intensity = baseIntensity + secondFluctuation + hourComponent + (Math.random() * 0.2 - 0.1);

        // Ensure non-negative and apply a cap
        intensity = Math.max(0.01, intensity); // Minimum 0.01
        intensity = Math.min(5.0, intensity);  // Maximum 5.0

        return intensity.toFixed(2); // Return as string with 2 decimal places
    }


    function updateClock() {
        const now = new Date();

        if (yearElem) {
            yearElem.textContent = now.getFullYear();
        }
        if (weekElem) {
            weekElem.textContent = String(getWeekNumber(now)).padStart(2, '0');
        }
        if (hoursElem) {
            hoursElem.textContent = String(now.getHours()).padStart(2, '0');
        }
        if (minutesElem) {
            minutesElem.textContent = String(now.getMinutes()).padStart(2, '0');
        }
        if (secondsElem) {
            secondsElem.textContent = String(now.getSeconds()).padStart(2, '0');
        }
        if (cosmicRayElem) {
            cosmicRayElem.textContent = calculateCosmicRayIntensity(now);
        }
    }

    // Check if primary clock elements exist before setting interval
    if (hoursElem && minutesElem && secondsElem) {
        updateClock(); // Initial call
        setInterval(updateClock, 1000); // Update every second
    } else {
        console.warn("Core clock elements (hours, minutes, seconds) not found. Clock will not update.");
    }

    // --- Event listener for window resize to adjust dialog positions (basic) ---
    window.addEventListener('resize', () => {
        document.querySelectorAll('.dialog-box').forEach(dialog => {
            const dialogWidth = dialog.offsetWidth;
            const dialogHeight = dialog.offsetHeight;
            // Recalculate max X and Y based on new window size
            const maxX = contentArea.offsetWidth - dialogWidth - 10;
            const maxY = contentArea.offsetHeight - dialogHeight - 10;
            let currentX = parseInt(dialog.style.left) || 0;
            let currentY = parseInt(dialog.style.top) || 0;

            // Ensure dialogs stay within the new bounds
            dialog.style.left = `${Math.min(Math.max(0, currentX), Math.max(0, maxX))}px`;
            dialog.style.top = `${Math.min(Math.max(0, currentY), Math.max(0, maxY))}px`;
        });
    });
});