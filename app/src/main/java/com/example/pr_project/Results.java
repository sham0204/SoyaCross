package com.example.pr_project;

import android.os.Bundle;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class Results extends AppCompatActivity {

    TextView resultTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_results);

        resultTextView = findViewById(R.id.result_text); // Ensure you have a TextView with this ID

        // Get result from Intent passed from MainActivity
        String result = getIntent().getStringExtra("model_output");

        // If no result is passed, show a default message
        if (result != null) {
            resultTextView.setText("Prediction Result:\n" + result);
        } else {
            resultTextView.setText("No result available.");
        }
    }
}
