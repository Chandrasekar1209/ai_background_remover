// frontend/src/App.js
import React, { useState } from 'react';
import Upload from './components/Upload';
import Preview from './components/Preview';
import { CirclePicker } from 'react-color';
import { uploadImage } from './services/api';
import './App.css';

function App() {
  const [originalImage, setOriginalImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedColor, setSelectedColor] = useState('#FFFFFF');
  const [customBackground, setCustomBackground] = useState(null);

  const handleDrop = async (acceptedFiles) => {
    setLoading(true);
    const file = acceptedFiles[0];
    
    try {
      // Display original image
      const reader = new FileReader();
      reader.onload = () => setOriginalImage(reader.result);
      reader.readAsDataURL(file);

      // Send to backend for processing
      const response = await uploadImage(file);
      setProcessedImage(`${response.config.baseURL}/${response.data.output_path}`);
    } catch (error) {
      console.error('Error processing image:', error);
      alert('Failed to process image');
    }
    setLoading(false);
  };

  const handleColorChange = (color) => {
    setSelectedColor(color.hex);
    setCustomBackground(null);
  };

  const handleCustomBackground = (file) => {
    const reader = new FileReader();
    reader.onload = () => setCustomBackground(reader.result);
    reader.readAsDataURL(file);
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = processedImage;
    link.download = 'background-removed.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="App">
      <h1>AI Background Remover</h1>
      
      <Upload onDrop={handleDrop} />
      
      {loading && <div className="loading">Processing...</div>}
      
      <div className="preview-container">
        {originalImage && (
          <div className="preview-section">
            <h3>Original</h3>
            <img src={originalImage} alt="Original" className="preview-image" />
          </div>
        )}
        
        {processedImage && (
          <div className="preview-section">
            <h3>Result</h3>
            <Preview 
              image={processedImage} 
              background={customBackground || selectedColor}
            />
            <div className="controls">
              <div className="color-picker">
                <h4>Background Color</h4>
                <CirclePicker 
                  color={selectedColor}
                  onChangeComplete={handleColorChange}
                />
              </div>
              
              <div className="custom-bg">
                <h4>Custom Background</h4>
                <input 
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleCustomBackground(e.target.files[0])}
                />
              </div>
              
              <button onClick={handleDownload} className="download-btn">
                Download Image
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;