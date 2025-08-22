package com.example.pr_project;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.*;

public class Helper {

    private static final String TAG = "ONNXHelper";
    private final OrtEnvironment env;

    public Helper() {
        env = OrtEnvironment.getEnvironment();
    }

    public OrtSession createModelSession(Context context, String modelName) throws OrtException {
        try {
            String modelPath = copyAssetToInternalStorage(context, modelName);
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            return env.createSession(modelPath, options);
        } catch (IOException e) {
            Log.e(TAG, "Failed to copy model file: " + modelName, e);
            throw new OrtException("Error copying model file: " + e.getMessage());
        } catch (OrtException e) {
            Log.e(TAG, "Failed to create ONNX session", e);
            throw e;
        }
    }

    public float[][] runInference(OrtSession session, float[] inputData) throws OrtException {
        long[] inputShape = new long[]{1, 224, 224, 3}; // NHWC format

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), inputShape)) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            String inputName = session.getInputNames().iterator().next();
            inputs.put(inputName, inputTensor);

            try (OrtSession.Result result = session.run(inputs)) {
                Object rawOutput = result.get(0).getValue();
                if (rawOutput instanceof float[][]) {
                    return (float[][]) rawOutput;
                } else if (rawOutput instanceof float[]) {
                    return new float[][]{(float[]) rawOutput};
                } else {
                    Log.e(TAG, "Unexpected output type: " + rawOutput.getClass().getSimpleName());
                    throw new OrtException("Unsupported output type: " + rawOutput.getClass().getSimpleName());
                }
            }
        } catch (OrtException e) {
            Log.e(TAG, "Inference failed", e);
            throw e;
        }
    }

    public String getPredictedClass(float[][] output, String[] classLabels) {
        // Assume output is [1, num_classes] or [num_classes]
        float[] scores;
        if (output.length == 1) {
            scores = output[0]; // Typical case: [1, num_classes]
        } else {
            scores = output[0]; // Fallback: assume first array if unexpected shape
            Log.w(TAG, "Unexpected output shape, using first array");
        }

        // Apply softmax to convert logits to probabilities
        float[] probabilities = softmax(scores);

        int maxIndex = 0;
        float maxScore = probabilities[0];

        // Find the index with the highest probability
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxScore) {
                maxScore = probabilities[i];
                maxIndex = i;
            }
        }

        // Build result string with class name and all probabilities
        StringBuilder result = new StringBuilder();
        if (maxIndex >= 0 && maxIndex < classLabels.length) {
            result.append("Predicted Class: ").append(classLabels[maxIndex])
                    .append(" (").append(String.format("%.2f%%", maxScore * 100)).append(")\n\n")
                    .append("All Probabilities:\n");
            for (int i = 0; i < probabilities.length && i < classLabels.length; i++) {
                result.append(classLabels[i]).append(": ")
                        .append(String.format("%.2f%%", probabilities[i] * 100)).append("\n");
            }
        } else {
            Log.e(TAG, "Predicted class index out of bounds: " + maxIndex);
            result.append("Unknown Class");
        }

        return result.toString();
    }

    private float[] softmax(float[] logits) {
        float[] probabilities = new float[logits.length];
        float maxLogit = logits[0];
        for (float logit : logits) {
            if (logit > maxLogit) maxLogit = logit;
        }

        float sum = 0.0f;
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = (float) Math.exp(logits[i] - maxLogit);
            sum += probabilities[i];
        }

        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }

        return probabilities;
    }

    private String copyAssetToInternalStorage(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            Log.d(TAG, "Model file already exists: " + file.getAbsolutePath());
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName);
             FileOutputStream fos = new FileOutputStream(file)) {
            byte[] buffer = new byte[4096];
            int length;
            while ((length = is.read(buffer)) > 0) {
                fos.write(buffer, 0, length);
            }
            fos.flush();
            Log.d(TAG, "Model file copied to: " + file.getAbsolutePath());
        } catch (IOException e) {
            Log.e(TAG, "Error copying asset: " + assetName, e);
            throw e;
        }

        return file.getAbsolutePath();
    }
}