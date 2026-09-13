import 'package:flutter/material.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final ort = OnnxRuntime();

  try {
    final session = await ort.createSessionFromAsset(
      'assets/caninecue_resnet18.onnx',
    );

    print('ONNX MODEL LOADED SUCCESSFULLY');
    print('Input names: ${session.inputNames}');
    print('Output names: ${session.outputNames}');

    await session.close();
  } catch (e) {
    print('ONNX MODEL ERROR: $e');
  }

  runApp(
    const MaterialApp(
      home: Scaffold(
        body: Center(
          child: Text('ONNX Test Complete'),
        ),
      ),
    ),
  );
}