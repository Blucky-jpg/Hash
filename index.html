import React, { useState, useEffect } from 'react';
import { Check, X, CheckCircle, ArrowLeft } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';

const hiraganaData = [
  {
    sectionName: "Vowels",
    characters: [
      { character: 'あ', romaji: 'a' },
      { character: 'い', romaji: 'i' },
      { character: 'う', romaji: 'u' },
      { character: 'え', romaji: 'e' },
    ]
  },
  {
    sectionName: "K-row",
    characters: [
      { character: 'か', romaji: 'ka' },
      { character: 'き', romaji: 'ki' },
      { character: 'く', romaji: 'ku' },
      { character: 'け', romaji: 'ke' },
    ]
  },
  {
    sectionName: "S-row",
    characters: [
      { character: 'さ', romaji: 'sa' },
      { character: 'し', romaji: 'shi' },
      { character: 'す', romaji: 'su' },
      { character: 'せ', romaji: 'se' },
    ]
  },
];

const LandingPage = ({ onSelectSection }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
      {hiraganaData.map((section) => (
        <Card key={section.sectionName} className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <CardTitle>{section.sectionName}</CardTitle>
            <CardDescription>{section.characters.length} characters</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2 mb-4">
              {section.characters.map((char) => (
                <span key={char.character} className="text-2xl">{char.character}</span>
              ))}
            </div>
            <Button onClick={() => onSelectSection(section)} className="w-full">
              Practice
            </Button>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

const LearningPage = ({ section, onReturn }) => {
  const [currentCharacter, setCurrentCharacter] = useState(null);
  const [options, setOptions] = useState([]);
  const [result, setResult] = useState(null);
  const [progress, setProgress] = useState(0);
  const [mistakes, setMistakes] = useState([]);
  const [isReviewMode, setIsReviewMode] = useState(false);
  const [answeredCharacters, setAnsweredCharacters] = useState(new Set());
  const [isAnswered, setIsAnswered] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);

  const resetLearning = () => {
    setProgress(0);
    setMistakes([]);
    setIsReviewMode(false);
    setAnsweredCharacters(new Set());
    setIsCompleted(false);
    newQuestion();
  };

  const newQuestion = () => {
    if (isReviewMode && mistakes.length === 0) {
      setIsCompleted(true);
      return;
    }

    let charactersPool = isReviewMode 
      ? mistakes 
      : section.characters.filter(char => !answeredCharacters.has(char.character));

    if (charactersPool.length === 0) {
      if (!isReviewMode) {
        setIsReviewMode(true);
        charactersPool = mistakes;
      } else {
        setIsCompleted(true);
        return;
      }
    }

    const randomIndex = Math.floor(Math.random() * charactersPool.length);
    const correct = charactersPool[randomIndex];
    setCurrentCharacter(correct);

    const wrongOptions = getRandomOptions(correct, 3);
    const allOptions = [correct, ...wrongOptions].sort(() => Math.random() - 0.5);
    setOptions(allOptions);
    setResult(null);
    setIsAnswered(false);
  };

  const getRandomOptions = (correct, count) => {
    const options = [];
    const allCharacters = hiraganaData.flatMap(section => section.characters);
    while (options.length < count) {
      const randomOption = allCharacters[Math.floor(Math.random() * allCharacters.length)];
      if (randomOption !== correct && !options.includes(randomOption)) {
        options.push(randomOption);
      }
    }
    return options;
  };

  const handleAnswer = (selected) => {
    if (isAnswered) return;
    setIsAnswered(true);

    if (selected === currentCharacter) {
      setResult('correct');
      if (!isReviewMode) {
        setProgress(prev => prev + 1);
        setAnsweredCharacters(prev => new Set(prev).add(currentCharacter.character));
      } else {
        setMistakes(prevMistakes => prevMistakes.filter(m => m !== currentCharacter));
      }
    } else {
      setResult('incorrect');
      if (!isReviewMode) {
        setMistakes(prevMistakes => [...prevMistakes, currentCharacter]);
      }
    }
  };

  useEffect(() => {
    resetLearning();
  }, [section]);

  useEffect(() => {
    if (isCompleted) {
      const timer = setTimeout(() => {
        onReturn();
      }, 2000); // Redirect after 2 seconds
      return () => clearTimeout(timer);
    }
  }, [isCompleted, onReturn]);

  if (isCompleted) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
        <div className="bg-white rounded-lg shadow-lg p-8 w-full max-w-md text-center">
          <h2 className="text-2xl font-bold mb-4">Section Completed!</h2>
          <p className="mb-4">Great job! You've finished the {section.sectionName} section.</p>
          <p>Redirecting to section selection...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white rounded-lg shadow-lg p-8 w-full max-w-md">
        <div className="flex justify-between items-center mb-4">
          <Button variant="outline" onClick={onReturn} className="flex items-center">
            <ArrowLeft className="mr-2" size={16} />
            Back to Sections
          </Button>
          <h2 className="text-xl font-bold">{section.sectionName}</h2>
        </div>
        <Progress 
          value={(progress / section.characters.length) * 100} 
          className="mb-4" 
        />
        {currentCharacter && (
          <>
            <div className="text-center mb-8">
              <span className="text-8xl font-bold">{currentCharacter.character}</span>
            </div>
            <div className="grid grid-cols-2 gap-4 mb-8">
              {options.map((option) => (
                <Button
                  key={option.romaji}
                  onClick={() => handleAnswer(option)}
                  className="text-lg py-3"
                  disabled={isAnswered}
                >
                  {option.romaji}
                </Button>
              ))}
            </div>
          </>
        )}
        {result && (
          <Alert variant={result === 'correct' ? 'default' : 'destructive'} className="mb-4">
            <AlertTitle className="flex items-center">
              {result === 'correct' ? (
                <Check className="mr-2" />
              ) : (
                <X className="mr-2" />
              )}
              {result === 'correct' ? 'Correct!' : 'Incorrect'}
            </AlertTitle>
            <AlertDescription>
              {result === 'correct'
                ? "Great job! Click 'Next Question' to continue."
                : `The correct answer was "${currentCharacter.romaji}". Click 'Next Question' to try again.`}
            </AlertDescription>
          </Alert>
        )}
        <div className="flex justify-between items-center">
          <Button onClick={newQuestion} className="w-full">
            {isAnswered ? 'Next Question' : 'Skip'}
          </Button>
        </div>
      </div>
    </div>
  );
};

const HiraganaLearningApp = () => {
  const [selectedSection, setSelectedSection] = useState(null);

  const handleSelectSection = (section) => {
    setSelectedSection(section);
  };

  const handleReturnToLanding = () => {
    setSelectedSection(null);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Hiragana Learning App</h1>
        </div>
      </header>
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {selectedSection ? (
          <LearningPage section={selectedSection} onReturn={handleReturnToLanding} />
        ) : (
          <LandingPage onSelectSection={handleSelectSection} />
        )}
      </main>
    </div>
  );
};

export default HiraganaLearningApp;
