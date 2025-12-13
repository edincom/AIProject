const username = document.getElementById('username-display').textContent;
let currentMode = 'teach';
let testState = 'idle';
let currentQuestion = '';
let currentExpectedAnswer = '';
let currentKeyPoints = [];
let modeChangeLocked = false;

let teachMessages = [];
let testMessages = [];
let chapterConversations = {};
let chapterHistories = {};
let currentChapter = null;

const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('sendBtn');
const modeOptions = document.querySelectorAll('.mode-option');
const welcomeText = document.getElementById('welcome-text');
const chapterList = document.getElementById('chapterList');

async function loadChapterHistory() {
    try {
        const response = await fetch(`/get_chapter_history?username=${encodeURIComponent(username)}`);
        const data = await response.json();
        chapterHistories = data.chapters || {};
        renderChapterList();
    } catch (error) {
        console.error('Error loading chapter history:', error);
    }
}

function renderChapterList() {
    chapterList.innerHTML = '';
    const chapters = Object.keys(chapterHistories);
    if (chapters.length === 0) {
        chapterList.innerHTML = '<p style="color: #999; font-size: 0.9rem;">Aucun historique</p>';
        return;
    }
    chapters.forEach(chapter => {
        const count = chapterHistories[chapter].length;
        const item = document.createElement('div');
        item.className = 'chapter-item';
        if (chapter === currentChapter) {
            item.classList.add('active');
        }
        item.innerHTML = `
            <div class="chapter-title">${escapeHtml(chapter)}</div>
            <div class="chapter-count">${count} interaction${count > 1 ? 's' : ''}</div>
        `;
        item.onclick = () => loadChapterConversation(chapter);
        chapterList.appendChild(item);
    });
}

function loadChapterConversation(chapter) {
    currentChapter = chapter;
    renderChapterList();
    chatMessages.innerHTML = '';
    const interactions = chapterHistories[chapter] || [];
    interactions.forEach(interaction => {
        addMessage(interaction.question, 'user');
        addMessage(interaction.answer, 'ai');
    });
    saveCurrentConversation();
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

modeOptions.forEach(option => {
    option.addEventListener('click', () => {
        if (modeChangeLocked && option.dataset.mode !== currentMode) {
            alert('⚠️ Vous devez d\'abord terminer le test en cours avant de changer de mode.');
            return;
        }
        saveCurrentConversation();
        modeOptions.forEach(opt => opt.classList.remove('active'));
        option.classList.add('active');
        currentMode = option.dataset.mode;
        testState = 'idle';
        currentQuestion = '';
        if (currentMode === 'test') {
            welcomeText.textContent = "Mode test : Entrez un sujet pour générer une question d'évaluation.";
            userInput.placeholder = "Ex: La Seconde Guerre mondiale";
        } else {
            welcomeText.textContent = "Je suis votre assistant d'apprentissage IA. Posez-moi des questions et je serai ravi de vous aider !";
            userInput.placeholder = "Posez votre question...";
        }
        loadConversation();
    });
});

function saveCurrentConversation() {
    const messages = Array.from(chatMessages.children).map(child => ({
        html: child.outerHTML,
        classes: child.className
    }));
    if (currentChapter) {
        chapterConversations[currentChapter] = messages;
        console.log(`💾 Saved to chapter "${currentChapter}":`, messages);
    }
    if (currentMode === 'teach') {
        teachMessages = messages;
    } else {
        testMessages = messages;
    }
}

function loadConversation() {
    chatMessages.innerHTML = '';
    const messages = currentMode === 'teach' ? teachMessages : testMessages;
    console.log(`📖 Loading ${currentMode} mode conversation:`, messages);
    if (messages.length === 0) {
        showWelcomeMessage();
    } else {
        messages.forEach(msg => {
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = msg.html;
            chatMessages.appendChild(tempDiv.firstChild);
        });
    }
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showWelcomeMessage() {
    const welcomeDiv = document.createElement('div');
    welcomeDiv.className = 'welcome-message';
    welcomeDiv.innerHTML = `
        <h2>👋 Bonjour <span id="welcome-username">${username}</span> !</h2>
        <p id="welcome-text">${currentMode === 'test'
            ? "Mode test : Entrez un sujet pour générer une question d'évaluation."
            : "Je suis votre assistant d'apprentissage IA. Posez-moi des questions et je serai ravi de vous aider !"}</p>
    `;
    chatMessages.appendChild(welcomeDiv);
}

function clearCurrentConversation() {
    if (currentMode === 'teach') {
        teachMessages = [];
    } else {
        testMessages = [];
    }
    currentChapter = null;
    loadConversation();
}

userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;
    console.log(`📤 Sending message in ${currentMode} mode:`, message);
    console.log(`📚 Current ${currentMode} history:`, currentMode === 'teach' ? teachMessages : testMessages);
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
        if (currentMode === 'teach') {
            teachMessages = teachMessages.filter(msg => !msg.classes.includes('welcome-message'));
        } else {
            testMessages = testMessages.filter(msg => !msg.classes.includes('welcome-message'));
        }
    }
    if (currentMode === 'teach') {
        await handleTeachMode(message);
    } else {
        await handleTestMode(message);
    }
    userInput.value = '';
    userInput.style.height = 'auto';
}

async function handleTeachMode(message) {
    addMessage(message, 'user');
    sendBtn.disabled = true;
    userInput.disabled = true;
    const aiMessageDiv = createMessageContainer('ai');
    const aiContentDiv = aiMessageDiv.querySelector('.message-content');
    try {
        const response = await fetch('/chat_api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                mode: 'teach',
                username: username
            })
        });
        if (!response.ok) {
            throw new Error('Erreur de connexion avec le serveur');
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const messages = buffer.split('\n\n');
            buffer = messages.pop();
            for (const message of messages) {
                if (message.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(message.substring(6));
                        if (data.error) {
                            aiContentDiv.innerHTML = `<span style="color:red;">${escapeHtml(data.error)}</span>`;
                        } else if (data.token) {
                            aiContentDiv.textContent += data.token;
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        }
                    } catch (err) {
                        console.error('Parse error:', err);
                    }
                }
            }
        }
        await loadChapterHistory();
    } catch (error) {
        aiContentDiv.innerHTML = `<span style="color:red;">Désolé, une erreur est survenue. Veuillez réessayer.</span>`;
        console.error('Error:', error);
    } finally {
        sendBtn.disabled = false;
        userInput.disabled = false;
        userInput.focus();
        saveCurrentConversation();
        console.log(`✅ Updated ${currentMode} history:`, teachMessages);
    }
}

async function handleTestMode(message) {
    if (testState === 'idle') {
        await generateQuestion(message);
    } else if (testState === 'waiting_for_answer') {
        await gradeAnswer(message);
    }
}

async function generateQuestion(criteria) {
    addMessage(`Sujet demandé : ${criteria}`, 'user');
    sendBtn.disabled = true;
    userInput.disabled = true;
    const loadingDiv = createMessageContainer('ai');
    const loadingContent = loadingDiv.querySelector('.message-content');
    loadingContent.innerHTML = '<div class="loading"><span></span><span></span><span></span></div> Génération de la question...';
    try {
        const response = await fetch('/test_api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                action: 'generate',
                criteria: criteria
            })
        });
        const data = await response.json();
        loadingDiv.remove();
        if (data.success) {
            currentQuestion = data.question;
            currentExpectedAnswer = data.expected_answer;
            currentKeyPoints = data.key_points;
            testState = 'waiting_for_answer';
            modeChangeLocked = true;
            const questionDiv = createMessageContainer('ai');
            const questionContent = questionDiv.querySelector('.message-content');
            questionContent.innerHTML = `
                <div class="test-question-box">
                    <h3>📝 Question d'évaluation</h3>
                    <p>${escapeHtml(currentQuestion)}</p>
                </div>
                <p style="margin-top: 1rem; color: #666;">Veuillez répondre à cette question dans le champ de saisie ci-dessous.</p>
            `;
            userInput.placeholder = "Écrivez votre réponse ici...";
        } else {
            throw new Error(data.error || 'Erreur lors de la génération de la question');
        }
    } catch (error) {
        loadingDiv.remove();
        const errorDiv = createMessageContainer('ai');
        const errorContent = errorDiv.querySelector('.message-content');
        errorContent.innerHTML = `<div class="error-message">Erreur : ${escapeHtml(error.message)}</div>`;
        testState = 'idle';
    } finally {
        sendBtn.disabled = false;
        userInput.disabled = false;
        userInput.focus();
    }
}

async function gradeAnswer(answer) {
    addMessage(answer, 'user');
    sendBtn.disabled = true;
    userInput.disabled = true;
    const loadingDiv = createMessageContainer('ai');
    const loadingContent = loadingDiv.querySelector('.message-content');
    loadingContent.innerHTML = '<div class="loading"><span></span><span></span><span></span></div> Évaluation en cours...';
    try {
        const response = await fetch('/test_api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                action: 'grade',
                question: currentQuestion,
                answer: answer,
                expected_answer: currentExpectedAnswer,
                key_points: currentKeyPoints, 
                username: username
            })
        });
        const data = await response.json();
        loadingDiv.remove();
        if (data.success) {
            displayGradingResult(data.grading_result);
            testState = 'idle';
            modeChangeLocked = false;
            currentQuestion = '';
            userInput.placeholder = "Entrez un nouveau sujet pour un autre test...";
        } else {
            throw new Error(data.error || 'Erreur lors de l\'évaluation');
        }
    } catch (error) {
        loadingDiv.remove();
        const errorDiv = createMessageContainer('ai');
        const errorContent = errorDiv.querySelector('.message-content');
        errorContent.innerHTML = `<div class="error-message">Erreur : ${escapeHtml(error.message)}</div>`;
        testState = 'idle';
    } finally {
        sendBtn.disabled = false;
        userInput.disabled = false;
        userInput.focus();
        saveCurrentConversation();
        console.log(`✅ Updated ${currentMode} history:`, testMessages);
    }
}

function displayGradingResult(data) {
    const resultDiv = createMessageContainer('ai');
    const resultContent = resultDiv.querySelector('.message-content');
    const scores = data.scores || {};
    
    resultContent.innerHTML = `
        <div class="grade-result">
            <div class="grade-header">
                <div>
                    <div class="grade-label">Note finale</div>
                    <div class="grade-score">${data.grade}/100</div>
                </div>
            </div>
            
            <div class="score-breakdown">
                <div class="score-item">
                    <div class="score-item-label">Points Clés</div>
                    <div class="score-item-value">${scores['Key Points'] || 0}/50</div>
                </div>
                <div class="score-item">
                    <div class="score-item-label">Correspondance Attendue</div>
                    <div class="score-item-value">${scores['Expected Match'] || 0}/30</div>
                </div>
                <div class="score-item">
                    <div class="score-item-label">Faits Incorrects</div>
                    <div class="score-item-value">${scores['Incorrect Facts'] || 0}/10</div>
                </div>
                <div class="score-item">
                    <div class="score-item-label">Structure</div>
                    <div class="score-item-value">${scores.Structure || 0}/10</div>
                </div>
            </div>
            
            <div class="advice-box">
                <h4>💡 Conseils pour s'améliorer</h4>
                <p>${escapeHtml(data.advice || 'Aucun conseil disponible')}</p>
            </div>
        </div>
    `;
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function createMessageContainer(type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = type === 'user' ? '👤' : '🤖';
    const content = document.createElement('div');
    content.className = 'message-content';
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageDiv;
}

function addMessage(text, type) {
    const messageDiv = createMessageContainer(type);
    const content = messageDiv.querySelector('.message-content');
    content.textContent = text;
    return messageDiv;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

sendBtn.addEventListener('click', sendMessage);

document.getElementById('clearChatBtn').addEventListener('click', () => {
    if (confirm('Êtes-vous sûr de vouloir effacer cette conversation ?')) {
        clearCurrentConversation();
    }
});

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !sendBtn.disabled) {
        e.preventDefault();
        sendMessage();
    }
});

async function loadHistory() {
    const modal = document.getElementById('historyModal');
    const historyDiv = document.getElementById("history");
    modal.style.display = 'block';
    historyDiv.innerHTML = "<p style='color:white; text-align:center;'>Chargement...</p>";
    try {
        const response = await fetch(`/history?username=${encodeURIComponent(username)}`);
        const data = await response.json();
        if (data.error) {
            historyDiv.innerHTML = `<p style="color:#ff6b6b; background:rgba(255,255,255,0.1); padding:1rem; border-radius:10px; text-align:center;">Erreur : ${escapeHtml(data.error)}</p>`;
            return;
        }
        if (!data.history || data.history.length === 0) {
            historyDiv.innerHTML = "<p style='color:white; background:rgba(255,255,255,0.1); padding:1rem; border-radius:10px; text-align:center;'>Aucun historique disponible.</p>";
            return;
        }
        let html = "<h3 style='color:white; text-align:center; margin-bottom:1.5rem; font-size:1.8rem;'>📜 Historique des questions</h3>";
        html += "<div style='background:rgba(255,255,255,0.95); padding:2rem; border-radius:20px; box-shadow:0 10px 40px rgba(0,0,0,0.3);'>";
        data.history.forEach((item, index) => {
            const gradeColor = item.grade >= 75 ? '#4caf50' : item.grade >= 50 ? '#ff9800' : '#f44336';
            html += `
                <div style='margin-bottom:1.5rem; padding:1.5rem; background:#f8f9fa; border-radius:15px; border-left:5px solid ${gradeColor}; box-shadow:0 2px 8px rgba(0,0,0,0.1);'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;'>
                        <span style='font-weight:bold; color:#667eea; font-size:1.1rem;'>Question ${index + 1}</span>
                        <span style='background:${gradeColor}; color:white; padding:0.5rem 1rem; border-radius:20px; font-weight:bold; font-size:1.2rem;'>${item.grade}%</span>
                    </div>
                    <div style='margin-bottom:0.8rem;'>
                        <strong style='color:#333;'>Question :</strong> 
                        <p style='color:#555; margin:0.5rem 0; line-height:1.6;'>${escapeHtml(item.question)}</p>
                    </div>
                    <div style='margin-bottom:0.8rem;'>
                        <strong style='color:#333;'>Réponse :</strong> 
                        <p style='color:#555; margin:0.5rem 0; line-height:1.6;'>${escapeHtml(item.answer)}</p>
                    </div>
                    ${item.advice ? `
                        <div style='background:#e3f2fd; padding:1rem; border-radius:8px; margin-top:1rem;'>
                            <strong style='color:#1976d2;'>💡 Conseils :</strong>
                            <p style='color:#1976d2; margin:0.5rem 0; line-height:1.6;'>${escapeHtml(item.advice)}</p>
                        </div>
                    ` : ''}
                    <button onclick="retest(${item.id}); closeHistory();" style='margin-top:1rem; padding:0.7rem 1.5rem; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:white; border:none; border-radius:25px; cursor:pointer; font-weight:600; transition:all 0.3s; box-shadow:0 4px 15px rgba(102,126,234,0.4);'>
                        🔁 Refaire cette question
                    </button>
                </div>
            `;
        });
        html += "</div>";
        historyDiv.innerHTML = html;
    } catch (err) {
        console.error(err);
        historyDiv.innerHTML = "<p style='color:#ff6b6b; background:rgba(255,255,255,0.1); padding:1rem; border-radius:10px; text-align:center;'>Erreur de connexion.</p>";
    }
}

async function retest(id) {
    try {
        const response = await fetch(`/retest?question_id=${id}&username=${encodeURIComponent(username)}`);
        const data = await response.json();
        if (data.error) {
            alert("Erreur : " + data.error);
            return;
        }
        currentMode = "test";
        modeOptions.forEach(opt => opt.classList.remove('active'));
        document.querySelector('[data-mode="test"]').classList.add('active');
        chatMessages.innerHTML = '';
        currentQuestion = data.question;
        testState = 'waiting_for_answer';
        modeChangeLocked = true;
        const questionDiv = createMessageContainer('ai');
        const questionContent = questionDiv.querySelector('.message-content');
        questionContent.innerHTML = `
            <div class="test-question-box">
                <h3>📝 Question d'évaluation (Re-test)</h3>
                <p>${escapeHtml(data.question)}</p>
            </div>
            <p style="margin-top: 1rem; color: #666;">Veuillez répondre à cette question dans le champ de saisie ci-dessous.</p>
        `;
        userInput.placeholder = "Écrivez votre réponse ici...";
        userInput.focus();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
        console.error(err);
        alert("Erreur de connexion.");
    }
}

function closeHistory() {
    document.getElementById('historyModal').style.display = 'none';
}

loadChapterHistory();
userInput.focus();