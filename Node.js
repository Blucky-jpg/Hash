// Sample data
const hiraganaData = [
  {
    sectionName: "Vowels",
    characters: [
      { character: "あ", romaji: "a" },
      { character: "い", romaji: "i" },
      { character: "う", romaji: "u" },
      { character: "え", romaji: "e" },
    ],
  },
  {
    sectionName: "K-row",
    characters: [
      { character: "か", romaji: "ka" },
      { character: "き", romaji: "ki" },
      { character: "く", romaji: "ku" },
      { character: "け", romaji: "ke" },
    ],
  },
  {
    sectionName: "S-row",
    characters: [
      { character: "さ", romaji: "sa" },
      { character: "し", romaji: "shi" },
      { character: "す", romaji: "su" },
      { character: "せ", romaji: "se" },
    ],
  },
];

const app = document.getElementById("app");

// Utility function to create a DOM element
function createElement(tag, className, content) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (content) element.textContent = content;
  return element;
}

// Function to render the landing page
function renderLandingPage() {
  const main = document.querySelector("main");
  main.innerHTML = ""; // Clear the current content

  const grid = createElement(
    "div",
    "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4"
  );

  hiraganaData.forEach((section) => {
    const card = createElement(
      "div",
      "card hover:shadow-lg transition-shadow"
    );
    const cardHeader = createElement("div", "card-header");
    const cardTitle = createElement("h2", "card-title", section.sectionName);
    const cardDescription = createElement(
      "p",
      "card-description",
      `${section.characters.length} characters`
    );

    cardHeader.appendChild(cardTitle);
    cardHeader.appendChild(cardDescription);

    const cardContent = createElement("div", "card-content");
    const charactersWrapper = createElement(
      "div",
      "flex flex-wrap gap-2 mb-4"
    );

    section.characters.forEach((char) => {
      const characterElement = createElement(
        "span",
        "text-2xl",
        char.character
      );
      charactersWrapper.appendChild(characterElement);
    });

    const practiceButton = createElement(
      "button",
      "w-full",
      "Practice"
    );

    practiceButton.addEventListener("click", () => {
      renderLearningPage(section);
    });

    cardContent.appendChild(charactersWrapper);
    cardContent.appendChild(practiceButton);

    card.appendChild(cardHeader);
    card.appendChild(cardContent);
    grid.appendChild(card);
  });

  main.appendChild(grid);
}

// Function to render the learning page
function renderLearningPage(section) {
  let currentCharacter = null;
  let options = [];
  let result = null;
  let progress = 0;
  let mistakes = [];
  let isReviewMode = false;
  let answeredCharacters = new Set();
  let isAnswered = false;
  let isCompleted = false;

  const main = document.querySelector("main");
  main.innerHTML = ""; // Clear the current content

  const container = createElement(
    "div",
    "flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4"
  );
  const card = createElement(
    "div",
    "bg-white rounded-lg shadow-lg p-8 w-full max-w-md"
  );

  const header = createElement("div", "flex justify-between items-center mb-4");
  const backButton = createElement("button", "flex items-center");
  backButton.textContent = "Back to Sections";
  backButton.addEventListener("click", renderLandingPage);

  const sectionTitle = createElement("h2", "text-xl font-bold", section.sectionName);

  header.appendChild(backButton);
  header.appendChild(sectionTitle);

  const progressBar = createElement("div", "progress mb-4");
  const progressFill = createElement("div", "progress-bar");
  progressBar.appendChild(progressFill);

  card.appendChild(header);
  card.appendChild(progressBar);

  function updateProgress() {
    progressFill.style.width = `${(progress / section.characters.length) * 100}%`;
  }

  function newQuestion() {
    if (isReviewMode && mistakes.length === 0) {
      isCompleted = true;
      setTimeout(renderLandingPage, 2000); // Redirect after 2 seconds
      return;
    }

    let charactersPool = isReviewMode
      ? mistakes
      : section.characters.filter((char) => !answeredCharacters.has(char.character));

    if (charactersPool.length === 0) {
      if (!isReviewMode) {
        isReviewMode = true;
        charactersPool = mistakes;
      } else {
        isCompleted = true;
        setTimeout(renderLandingPage, 2000);
        return;
      }
    }

    const randomIndex = Math.floor(Math.random() * charactersPool.length);
    const correct = charactersPool[randomIndex];
    currentCharacter = correct;

    const wrongOptions = getRandomOptions(correct, 3);
    options = [correct, ...wrongOptions].sort(() => Math.random() - 0.5);
    result = null;
    isAnswered = false;
    renderQuestion();
  }

  function getRandomOptions(correct, count) {
    const options = [];
    const allCharacters = hiraganaData.flatMap((section) => section.characters);
    while (options.length < count) {
      const randomOption =
        allCharacters[Math.floor(Math.random() * allCharacters.length)];
      if (randomOption !== correct && !options.includes(randomOption)) {
        options.push(randomOption);
      }
    }
    return options;
  }

  function handleAnswer(selected) {
    if (isAnswered) return;
    isAnswered = true;

    if (selected === currentCharacter) {
      result = "correct";
      if (!isReviewMode) {
        progress++;
        answeredCharacters.add(currentCharacter.character);
      } else {
        mistakes = mistakes.filter((m) => m !== currentCharacter);
      }
    } else {
      result = "incorrect";
      if (!isReviewMode) {
        mistakes.push(currentCharacter);
      }
    }

    renderQuestion();
  }

  function renderQuestion() {
    const questionWrapper = createElement("div", "text-center mb-8");
    const characterElement = createElement(
      "span",
      "text-8xl font-bold",
      currentCharacter.character
    );

    questionWrapper.appendChild(characterElement);

    const optionsWrapper = createElement("div", "grid grid-cols-2 gap-4 mb-8");
    options.forEach((option) => {
      const optionButton = createElement(
        "button",
        "text-lg py-3",
        option.romaji
      );
      optionButton.disabled = isAnswered;
      optionButton.addEventListener("click", () => handleAnswer(option));
      optionsWrapper.appendChild(optionButton);
    });

    const alert = createElement(
      "div",
      `alert ${result === "correct" ? "alert-default" : "alert-destructive"} mb-4`
    );
    const alertTitle = createElement("div", "alert-title");
    const alertIcon = createElement("span");
    const alertDescription = createElement("p", "alert-description");

    if (result) {
      if (result === "correct") {
        alertIcon.textContent = "✔️";
        alertDescription.textContent = "Great job! Click 'Next Question' to continue.";
      } else {
        alertIcon.textContent = "❌";
        alertDescription.textContent = `The correct answer was "${currentCharacter.romaji}". Click 'Next Question' to try again.`;
      }
      alertTitle.appendChild(alertIcon);
      alertTitle.appendChild(createElement("span", null, result === "correct" ? "Correct!" : "Incorrect"));
      alert.appendChild(alertTitle);
      alert.appendChild(alertDescription);
    }

    const buttonWrapper = createElement("div", "flex justify-between items-center");
    const nextButton = createElement("button", "w-full", isAnswered ? "Next Question" : "Skip");
    nextButton.addEventListener("click", newQuestion);

    buttonWrapper.appendChild(nextButton);

    card.innerHTML = ""; // Clear previous content
    card.appendChild(header);
    card.appendChild(progressBar);
    card.appendChild(questionWrapper);
    card.appendChild(optionsWrapper);
    if (result) {
      card.appendChild(alert);
    }
    card.appendChild(buttonWrapper);
    updateProgress();
  }

  container.appendChild(card);
  main.appendChild(container);
  newQuestion();
}

// Initialize the app
renderLandingPage();
