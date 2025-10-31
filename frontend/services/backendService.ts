// Helper to convert data URL to Blob
const dataURLtoBlob = (dataurl: string): Blob => {
    const arr = dataurl.split(',');
    if (arr.length < 2) {
        throw new Error("Invalid data URL");
    }
    const mimeMatch = arr[0].match(/:(.*?);/);
    if (!mimeMatch) {
        throw new Error("Could not parse MIME type from data URL");
    }
    const mime = mimeMatch[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while(n--){
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], {type:mime});
}

const API_BASE_URL = 'http://localhost:8000/api';

export const generateMask = async (imageBase64: string, prompt: string): Promise<string> => {
    const formData = new FormData();
    // The backend expects a file named 'image.png'
    formData.append('image', dataURLtoBlob(imageBase64), 'image.png');
    formData.append('prompt', prompt);

    const response = await fetch(`${API_BASE_URL}/segment`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to generate mask: ${response.statusText} - ${errorText}`);
    }

    const result = await response.json();

    if (!result.success || !result.data?.mask) {
        throw new Error(result.error || "Backend failed to generate mask.");
    }

    // The backend returns a black and white mask.
    return result.data.mask;
};

export const generateEdit = async (
  imageBase64: string, 
  maskBase64: string, 
  prompt: string,
  inferenceSteps: number,
  seed: number
): Promise<string> => {
    const formData = new FormData();
    formData.append('image', dataURLtoBlob(imageBase64), 'image.png');
    formData.append('mask', dataURLtoBlob(maskBase64), 'mask.png');
    formData.append('prompt', prompt);
    formData.append('inference_steps', String(inferenceSteps));
    formData.append('seed', String(seed));

    const response = await fetch(`${API_BASE_URL}/inpaint`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to generate edited image: ${response.statusText} - ${errorText}`);
    }
    
    const result = await response.json();

    if (!result.success || !result.data?.result_image) {
        throw new Error(result.error || "Backend failed to generate edited image.");
    }

    return result.data.result_image;
};
