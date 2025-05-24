// Wait for the HTML document to be fully loaded and parsed before executing the script
document.addEventListener('DOMContentLoaded', () => {
    // Get references to key DOM elements
    // const contentArea = document.getElementById('contentArea'); // Area where dialogs will appear
    const dialogTemplate = document.getElementById('dialogTemplate'); // HTML template for dialogs
    const container = document.querySelector('.container'); // <<--- NEW: Get reference to the main container

    // Initialize z-index counter for dialog stacking.
    // Starts above the main logo's z-index (100) to ensure dialogs can come to the front.
    let highestZIndex = 101;

    // Configuration for dialog spawning
    const MAX_DIALOGS_TO_SPAWN = 10; // Maximum number of dialogs that will be spawned in total
    let spawnedDialogsCount = 0;   // Counter for how many dialogs have been spawned

    // --- Global variable to hold dialogs data once fetched ---
    let DIALOGS_DATA_FROM_JSON = []; // Will be populated by fetch

    // --- Function to initialize and shuffle available dialog indices ---
    function initializeAvailableDialogs() {
        if (DIALOGS_DATA_FROM_JSON.length === 0) {
            console.warn("Dialogs data is not loaded yet or is empty. Cannot initialize.");
            return;
        }
        availableDialogIndices = DIALOGS_DATA_FROM_JSON.map((_, index) => index);
        for (let i = availableDialogIndices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [availableDialogIndices[i], availableDialogIndices[j]] = [availableDialogIndices[j], availableDialogIndices[i]];
        }
    }

    // Data for the dialogs. Each object represents one dialog.
    // 'id' is for potential future reference, 'title', 'width', and 'content' (HTML string) define the dialog.
    const dialogsData = [
        {
            id: 1,
            title: '💖 宇宙卵团子巡演',
            width: '340px',
            content: `<img src="https://via.placeholder.com/300x180/b4f8c8/333333?text=宇宙卵团子巡演图" alt="宇宙卵团子巡演"><p>ZUTOMAYO FACTORY「宇宙の卵子 DORODANGO」巡演开始！更多信息请查看官网。</p>`,
            yesLink: 'https://zutomayo.net/tour2024_dorodango/' // Example Link
        },
        {
            id: 2,
            title: '📅 3.29-5.18 竞技场之旅',
            width: '360px',
            content: `<img src="https://via.placeholder.com/320x150/a0e7e5/333333?text=竞技场之旅图" alt="竞技场之旅"><p>ZUTOMAYO 竞技场之旅 2024「本格的に」即将举行。日程: 3月29日 - 5月18日</p>`,
            yesLink: 'https://zutomayo.net/arena2024/' // Example Link
        },
        {
            id: 3,
            title: '📢 News 2025.05.18',
            width: '300px',
            content: `<p><strong>"YAKI YAK" 师父和师父</strong></p><p>新曲发布！详情请关注后续公告。</p>`
            // No yesLink, so "Yes" button might do nothing or be hidden/disabled
        },
        {
            id: 4,
            title: '💿 1st ZUTOMAYO',
            width: '320px',
            content: `<img src="https://via.placeholder.com/280x200/d7b0ff/333333?text=潜潜話专辑图" alt="潜潜話专辑"><p>首张专辑「潜潜話」好评发售中！探索ZUTOMAYO的音乐世界。</p>`,
            yesLink: 'https://store.zutomayo.com/products/detail/15' // Example Link
        },
        {
            id: 5,
            title: 'MV发布 5/22 21:00',
            width: '350px',
            content: `<p><strong>你能和 Cream 一起来看我吗？</strong></p><p>新MV将于 5月22日 21:00 (JST) 发布！敬请期待！不要错过！</p>`,
            yesLink: 'https://www.youtube.com/@ZUTOMAYO' // Example Link
        },
        {
            id: 6,
            title: '✨ 特别通知 ✨',
            width: '310px',
            content: `<p>感谢大家一直以来的支持！</p><p>未来将有更多精彩内容，请保持关注官方动态！</p>`
            // No yesLink for this one, maybe only "OK" (No button) is needed.
        },
        {
            id: 7,
            title: '🎶 新歌试听片段',
            width: '330px',
            content: `<p>最新单曲片段抢先听！</p><p>感受ZUTOMAYO的独特魅力。</p><img src="https://via.placeholder.com/290x100/f9c5d1/333333?text=新歌试听图" alt="新歌试听">`
            // No yesLink, "Yes" button could be "Listen More" if you had a link
        }
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
        const dialogClone = dialogTemplate.content.cloneNode(true);
        const dialogBox = dialogClone.querySelector('.dialog-box');
        const titleElement = dialogBox.querySelector('.dialog-title');
        const contentElement = dialogBox.querySelector('.dialog-content');
        const closeButton = dialogBox.querySelector('.dialog-close-button');

        // --- Get references to action buttons ---
        const actionButtons = dialogBox.querySelectorAll('.dialog-action-button');
        const yesButton = actionButtons[0]; // Assuming "はい (Y)" is the first
        const noButton = actionButtons[1];  // Assuming "いいえ (N)" is the second

        titleElement.textContent = data.title || 'Untitled Dialog';
        contentElement.innerHTML = data.content || '<p>No content.</p>';

        let dialogWidth = parseInt(data.width) || 300;
        let dialogHeight = 150; // Base height for positioning, content will determine actual

        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight; // Get viewport height as well

        // --- NEW LOGIC for Small Screens ---
        if (viewportWidth < 480) { // Example breakpoint for "very small"
            // For very small screens, drastically reduce default/parsed dialog width
            dialogWidth = Math.min(dialogWidth, viewportWidth * 0.85, 200); // e.g., 85% of viewport, or max 200px
            // Consider a minimum width too, e.g., 100px, so it's not invisibly small
            dialogWidth = Math.max(dialogWidth, 120);
        } else if (viewportWidth < 768) {
            dialogWidth = Math.min(dialogWidth, viewportWidth * 0.9, 280);
            dialogWidth = Math.max(dialogWidth, 150);
        } else {
            // For larger screens, ensure it doesn't exceed a reasonable max or CSS max-width
            const cssMaxWidth = parseFloat(getComputedStyle(dialogBox).maxWidth);
            if (!isNaN(cssMaxWidth) && cssMaxWidth < dialogWidth) {
                dialogWidth = cssMaxWidth;
            }
            dialogWidth = Math.min(dialogWidth, viewportWidth * 0.7); // Max 70% of viewport width on large screens
        }
        dialogBox.style.width = `${dialogWidth}px`;


        // Positioning:
        // Allow dialogs to appear even if contentArea is tiny or not fully defined yet.
        // Position relative to viewport initially, then adjust if needed.
        // For initial placement, use a simpler logic if strict bounds are not required.

        const containerRect = container.getBoundingClientRect();
        // Need to account for .container's padding-top if positioning from its 0,0
        const containerPaddingTop = parseFloat(getComputedStyle(container).paddingTop) || 0;

        let initialX, initialY;

        // If container is very small, fallback to viewport-relative positioning near top-left
        if (containerRect.width < dialogWidth || containerRect.height < dialogHeight) {
            initialX = 10 + Math.random() * 20;
            initialY = (document.querySelector('.main-logo')?.offsetHeight || 90) + 10 + Math.random() * 20;
            // Make sure it's within viewport if container is truly messed up
            initialX = Math.max(5, Math.min(initialX, viewportWidth - dialogWidth - 5));
            initialY = Math.max(5, Math.min(initialY, window.innerHeight - dialogHeight - 5));
        } else {
            // Random positioning within the .container bounds
            // Max X considers the full width of .container
            const maxX = Math.max(5, containerRect.width - dialogWidth - 5);
            // Max Y considers .container height, starting below its padding-top
            const maxY = Math.max(5, (containerRect.height - containerPaddingTop) - dialogHeight - 5);

            initialX = 5 + Math.random() * maxX;
            // initialY starts below the container's top padding (where the logo is)
            initialY = containerPaddingTop + 5 + (Math.random() * maxY);
        }

        dialogBox.style.left = `${initialX}px`;
        dialogBox.style.top = `${initialY}px`; // This is relative to .container's top edge now

        makeDraggable(dialogBox); // Pass dialogBox itself

        closeButton.addEventListener('click', () => dialogBox.remove());
        dialogBox.addEventListener('mousedown', () => {
            dialogBox.style.zIndex = ++highestZIndex;
            document.querySelectorAll('.dialog-box.active').forEach(el => el.classList.remove('active'));
            dialogBox.classList.add('active');
        }, true);


        // --- CONFIGURE ACTION BUTTONS ---
        if (noButton) {
            noButton.addEventListener('click', () => {
                dialogBox.remove(); // "No" button closes the dialog
            });
        } else {
            console.warn("No button not found for dialog:", data.title);
        }

        if (yesButton) {
            if (data.yesLink) {
                yesButton.addEventListener('click', () => {
                    window.open(data.yesLink, '_blank'); // Open link in a new tab
                    // Optionally, close the dialog after clicking "Yes"
                    // dialogBox.remove();
                });
            } else {
                // No yesLink provided for this dialog.
                // Option 1: Disable the "Yes" button
                // yesButton.disabled = true;
                // yesButton.style.opacity = "0.5";
                // yesButton.style.cursor = "not-allowed";

                // Option 2: Hide the "Yes" button if no link
                // yesButton.style.display = 'none';

                // Option 3: Make "Yes" button also close the dialog (acting like an "OK")
                yesButton.textContent = 'OK'; // Change text if you want
                noButton.style.display = 'none';
                yesButton.addEventListener('click', () => {
                    dialogBox.remove();
                });
            }
        } else {
            console.warn("Yes button not found for dialog:", data.title);
        }

        // If only one button is desired (e.g., only "OK" which is the "No" button functionality)
        // and the other is hidden/disabled:
        const dialogButtonsContainer = dialogBox.querySelector('.dialog-buttons');
        if (yesButton && yesButton.style.display === 'none' && noButton) {
            // If Yes is hidden, and No exists, center No button
            dialogButtonsContainer.style.textAlign = 'center'; // Or adjust specific button margins
        } else if (noButton && noButton.style.display === 'none' && yesButton) {
            // If No is hidden, and Yes exists, center Yes button
            dialogButtonsContainer.style.textAlign = 'center';
        }


        container.appendChild(dialogBox); // <<--- APPEND TO .container
        spawnedDialogsCount++;
        console.log(`Spawned ${spawnedDialogsCount} / ${MAX_DIALOGS_TO_SPAWN} dialogs (ID: ${data.id})...`);
        return dialogBox;
    }

    // --- Function to make a dialog element draggable ---
    function makeDraggable(element) {
        const titleBar = element.querySelector('.dialog-title-bar');
        let offsetX, offsetY, isDragging = false;
        // The 'container' is the new boundary parent for dragging.
        // Its getBoundingClientRect() gives viewport-relative coords.
        // element.offsetLeft/Top are relative to its offsetParent (which is now .container).

        titleBar.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('dialog-close-button')) return;
            isDragging = true;

            // Offset of mouse from top-left of draggable element
            offsetX = e.clientX - element.getBoundingClientRect().left;
            offsetY = e.clientY - element.getBoundingClientRect().top;
            // Alternatively, if element.offsetLeft/Top are reliable relative to .container:
            // offsetX = e.pageX - element.offsetLeft; // pageX relative to document
            // offsetY = e.pageY - element.offsetTop;  // pageY relative to document

            element.style.zIndex = ++highestZIndex;
            titleBar.style.cursor = 'grabbing';
            document.querySelectorAll('.dialog-box.active').forEach(el => el.classList.remove('active'));
            element.classList.add('active');
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            const containerRect = container.getBoundingClientRect();
            // New absolute position of element's top-left in viewport coordinates
            let newViewportX = e.clientX - offsetX;
            let newViewportY = e.clientY - offsetY;

            // Convert to position relative to .container for style.left/top
            let newStyleLeft = newViewportX - containerRect.left;
            let newStyleTop = newViewportY - containerRect.top;

            // --- DRAG CONSTRAINTS (Relative to .container) ---
            const elemWidth = element.offsetWidth;
            const elemHeight = element.offsetHeight;

            // Option: Constrain within .container, allowing title bar to be visible
            // This means newStyleLeft/Top are the values we constrain
            const minX = 0 - elemWidth + titleBar.offsetWidth; // Allow part of window off left
            const maxX = containerRect.width - titleBar.offsetWidth;   // Allow part of window off right
            const minY = 0; // Can go to top edge of .container
            const maxY = containerRect.height - titleBar.offsetHeight; // Title bar visible at bottom

            newStyleLeft = Math.max(minX, Math.min(newStyleLeft, maxX));
            newStyleTop = Math.max(minY, Math.min(newStyleTop, maxY));

            // Prevent from going above the main logo area (visual constraint)
            const logoHeight = (document.querySelector('.main-logo')?.offsetHeight || 0) + 10; // Approx height + margin
            newStyleTop = Math.max(newStyleTop, logoHeight);


            element.style.left = `${newStyleLeft}px`;
            element.style.top = `${newStyleTop}px`;
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                titleBar.style.cursor = 'grab';
                // element.classList.remove('active'); // Keep active for visual feedback of last touch
            }
        });
    }

    // --- Function to spawn the next dialog from the available list ---
    function spawnNextDialog() {
        if (DIALOGS_DATA_FROM_JSON.length === 0) {
            console.log("Waiting for dialog data to load...");
            return;
        }
        if (spawnedDialogsCount >= MAX_DIALOGS_TO_SPAWN || availableDialogIndices.length === 0) {
            console.log(`Spawned ${spawnedDialogsCount} dialogs. Limit reached or no unique dialogs left. Stopping.`);
            if (dialogCreationInterval) clearInterval(dialogCreationInterval);
            return;
        }
        const nextDialogDataOriginalIndex = availableDialogIndices.shift();
        const dialogDataToShow = DIALOGS_DATA_FROM_JSON[nextDialogDataOriginalIndex];
        if (dialogDataToShow) {
            createDialog(dialogDataToShow);
            // spawnedDialogsCount++; // Now incremented inside createDialog
        } else {
            console.warn("Attempted to get dialog data that does not exist.");
        }
    }

    // --- Function to start the dialog spawning process ---
    function startDialogSystem() {
        if (DIALOGS_DATA_FROM_JSON.length > 0) {
            initializeAvailableDialogs(); // Shuffle the newly loaded data
            spawnNextDialog(); // Spawn the first dialog immediately

            setTimeout(() => {
                if (spawnedDialogsCount < MAX_DIALOGS_TO_SPAWN && availableDialogIndices.length > 0) {
                    spawnNextDialog();
                }
            }, 1200);

            dialogCreationInterval = setInterval(spawnNextDialog, 2200);
        } else {
            console.log("No dialog data loaded to spawn.");
        }
    }

    // --- FETCH DIALOG DATA ---
    fetch('data/dialogs/dialogs-data.json') // Adjust path if you placed it elsewhere (e.g., 'js/dialogs-data.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            DIALOGS_DATA_FROM_JSON = data; // Store the fetched data globally
            console.log("Dialogs data loaded successfully:", DIALOGS_DATA_FROM_JSON);
            startDialogSystem(); // Now that data is loaded, start spawning dialogs
        })
        .catch(error => {
            console.error("Could not load dialogs data:", error);
            // You could display an error message to the user here if critical
        });


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
    // --- MODIFIED Window Resize Listener ---
    window.addEventListener('resize', () => {
        const viewportWidth = window.innerWidth;
        const containerRect = container.getBoundingClientRect(); // Use .container for bounds
        const containerPaddingTop = parseFloat(getComputedStyle(container).paddingTop) || 0;
        const logoHeight = (document.querySelector('.main-logo')?.offsetHeight || 0) + 10;

        document.querySelectorAll('.dialog-box').forEach(dialog => {
            let dialogWidth = dialog.offsetWidth;
            // const dialogHeight = dialog.offsetHeight; // Not strictly needed if not constraining vertically on resize

            // Re-apply width constraint from CSS if necessary or adapt to viewport
            if (viewportWidth < 480) {
                dialogWidth = Math.min(dialog.offsetWidth, viewportWidth * 0.85, 200);
                dialogWidth = Math.max(dialogWidth, 120);
            } else if (viewportWidth < 768) {
                dialogWidth = Math.min(dialog.offsetWidth, viewportWidth * 0.9, 280);
                dialogWidth = Math.max(dialogWidth, 150);
            } else {
                const cssMaxWidth = parseFloat(getComputedStyle(dialog).maxWidth);
                if (!isNaN(cssMaxWidth) && cssMaxWidth < dialog.offsetWidth) { // Use dialog.offsetWidth for comparison
                    dialogWidth = cssMaxWidth;
                }
                // Cap at 70% of viewport on large screens
                dialogWidth = Math.min(dialogWidth, viewportWidth * 0.7);
            }
            dialog.style.width = `${dialogWidth}px`;


            // --- REMOVE OR RELAX BOUNDARY CONSTRAINTS ON RESIZE ---
            const dialogHeight = dialog.offsetHeight;

            // --- REPOSITIONING LOGIC ON RESIZE (Relative to .container) ---
            let currentStyleLeft = parseFloat(dialog.style.left) || 0;
            let currentStyleTop = parseFloat(dialog.style.top) || 0;

            const minX = 0 - dialogWidth + (dialog.querySelector('.dialog-title-bar')?.offsetWidth || 30);
            const maxX = containerRect.width - (dialog.querySelector('.dialog-title-bar')?.offsetWidth || 30);
            const minY = logoHeight; // Don't go above logo
            const maxY = containerRect.height - (dialog.querySelector('.dialog-title-bar')?.offsetHeight || 20);

            currentStyleLeft = Math.max(minX, Math.min(currentStyleLeft, maxX));
            currentStyleTop = Math.max(minY, Math.min(currentStyleTop, maxY));

            // If it's WAY off after resize (e.g. container shrank a lot)
            if (currentStyleLeft > containerRect.width - 30) {
                currentStyleLeft = Math.max(minX, containerRect.width - dialogWidth);
            }
            if (currentStyleTop > containerRect.height - 20) {
                currentStyleTop = Math.max(minY, containerRect.height - dialogHeight);
            }


            dialog.style.left = `${currentStyleLeft}px`;
            dialog.style.top = `${currentStyleTop}px`;
        });
    });
});