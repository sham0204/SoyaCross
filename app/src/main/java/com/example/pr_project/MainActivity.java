package com.example.pr_project;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    private Button btnPicture, btnGallery, btnRunInference;
    private ImageView imageView;
    private OrtSession session;
    private Helper helper;
    private Bitmap selectedBitmap;
    private OrtEnvironment env;

    private static final int REQUEST_CODE_CAMERA = 22;

    // Define class labels (replace with your model's actual class labels)
    private static final String[] classLabels = {
            "Bacterial Postule ","Frogeye Leaf", "Healthy", "Rust", "Sudden Death Syndrome", "Target Leaf Spot", "Yellow Mosaic" // Update with actual labels
    };

    private ActivityResultLauncher<Intent> galleryResultLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        env = OrtEnvironment.getEnvironment();
        initUI();
        registerGalleryResult();

        helper = new Helper();

        try {
            session = helper.createModelSession(this, "model.onnx");
            Toast.makeText(this, "Model loaded successfully!", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Log.e(TAG, "Error loading model", e);
            Toast.makeText(this, "Model load failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }

        btnGallery.setOnClickListener(v -> openGallery());
        btnPicture.setOnClickListener(v -> openCamera());
        btnRunInference.setOnClickListener(v -> {
            if (selectedBitmap == null) {
                Toast.makeText(this, "Please select or capture an image!", Toast.LENGTH_SHORT).show();
                return;
            }
            String result = runInferenceOnImage();
            Intent intent = new Intent(MainActivity.this, Results.class);
            intent.putExtra("model_output", result);
            startActivity(intent);
        });
    }

    private void initUI() {
        btnPicture = findViewById(R.id.btncamera_id);
        btnGallery = findViewById(R.id.button2);
        btnRunInference = findViewById(R.id.Button);
        imageView = findViewById(R.id.imageview1);
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        galleryResultLauncher.launch(intent);
    }

    private void openCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(cameraIntent, REQUEST_CODE_CAMERA);
    }

    private void registerGalleryResult() {
        galleryResultLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        try {
                            Uri imgUri = result.getData().getData();
                            selectedBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imgUri);
                            imageView.setImageBitmap(selectedBitmap);
                        } catch (Exception e) {
                            Log.e(TAG, "Gallery image load error", e);
                            Toast.makeText(this, "Image load error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                        }
                    }
                });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_CODE_CAMERA && resultCode == RESULT_OK && data != null) {
            selectedBitmap = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(selectedBitmap);
        }
    }

    private String runInferenceOnImage() {
        try {
            float[] inputData = preprocessImage(selectedBitmap);
            float[][] output = helper.runInference(session, inputData);
            return processOutput(output);
        } catch (OrtException e) {
            Log.e(TAG, "Error during inference", e);
            Toast.makeText(this, "Inference failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
            return "Error: " + e.getMessage();
        }
    }

    private float[] preprocessImage(Bitmap bitmap) {
        final int IMG_SIZE = 224; // same as in your Python code

        // Step 1: Resize the image to (224, 224)
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMG_SIZE, IMG_SIZE, true);

        // Step 2: Get pixel data
        int width = resizedBitmap.getWidth();
        int height = resizedBitmap.getHeight();
        int[] pixels = new int[width * height];
        resizedBitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        // Step 3: Normalize and reshape to [1, 224, 224, 3]
        float[] inputData = new float[1 * height * width * 3];
        int index = 0;
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];

            // Extract RGB components and normalize to [0, 1]
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;

            inputData[index++] = r;
            inputData[index++] = g;
            inputData[index++] = b;
        }

        return inputData;
    }


    private String processOutput(float[][] output) {
        // Delegate to Helper to compute class name and percentages
        return helper.getPredictedClass(output, classLabels);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                Log.e(TAG, "Error closing session", e);
            }
        }
    }
}