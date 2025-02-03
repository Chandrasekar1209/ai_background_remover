import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1',
});

export const uploadImage = async (file) => {
  try {
    // Upload file
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/upload', formData);
    const taskId = response.data.task_id;

    console.log("Upload successful, task ID:", taskId);

    // Polling function to check task status
    const checkTaskStatus = async () => {
      const taskResponse = await api.get(`/task/${taskId}`);
      console.log("Task Status:", taskResponse.data);

      if (taskResponse.data.status === "completed") {
        console.log("Processed Image:", taskResponse.data.processed_image);
        alert("Processing done! Image available at: " + taskResponse.data.processed_image);
      } else if (taskResponse.data.status === "failed") {
        alert("Image processing failed: " + taskResponse.data.error);
      } else {
        // Retry after 2 seconds
        setTimeout(checkTaskStatus, 2000);
      }
    };

    checkTaskStatus();
  } catch (error) {
    console.error("Error uploading file:", error);
    alert("Upload failed: " + error.message);
  }
};
