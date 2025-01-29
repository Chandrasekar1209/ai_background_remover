// frontend/src/components/Upload.jsx
import React from 'react';
import { useDropzone } from 'react-dropzone';

export default function Upload({ onDrop }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: 'image/*'
  });

  return (
    <div {...getRootProps()} style={{ 
      padding: '2rem', 
      border: '2px dashed #aaa',
      textAlign: 'center'
    }}>
      <input {...getInputProps()} />
      {isDragActive ? (
        <p>Drop images here...</p>
      ) : (
        <p>Drag & drop images, or click to select</p>
      )}
    </div>
  );
}