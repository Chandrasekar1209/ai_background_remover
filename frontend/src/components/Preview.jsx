// frontend/src/components/Preview.jsx
import React from 'react';

const Preview = ({ image, background }) => {
  const previewStyle = {
    background: background.startsWith('#') ? background : `url(${background})`,
    backgroundSize: 'cover',
    padding: '1rem',
  };

  return (
    <div style={previewStyle}>
      <img 
        src={image} 
        alt="Processed" 
        style={{ 
          maxWidth: '500px', 
          maxHeight: '400px',
          mixBlendMode: 'multiply' 
        }} 
      />
    </div>
  );
};

export default Preview;