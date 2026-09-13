package com.example.caninecue

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.ByteArrayOutputStream

class MainActivity : FlutterActivity() {

    private val CHANNEL = "caninecue/video"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            CHANNEL
        ).setMethodCallHandler { call, result ->

            when (call.method) {

                "extractFrames" -> {
                    val videoPath = call.argument<String>("videoPath")
                    val frameCount = call.argument<Int>("frameCount") ?: 30

                    if (videoPath == null) {
                        result.error(
                            "INVALID_PATH",
                            "Video path is missing.",
                            null
                        )
                        return@setMethodCallHandler
                    }

                    try {
                        val frames = extractFrames(
                            videoPath,
                            frameCount
                        )

                        result.success(frames)

                    } catch (e: Exception) {
                        result.error(
                            "FRAME_EXTRACTION_ERROR",
                            e.message,
                            null
                        )
                    }
                }

                else -> {
                    result.notImplemented()
                }
            }
        }
    }

    private fun extractFrames(
        videoPath: String,
        frameCount: Int
    ): List<ByteArray> {

        val retriever = MediaMetadataRetriever()
        val frames = mutableListOf<ByteArray>()

        try {
            retriever.setDataSource(this, Uri.parse(videoPath))

            val durationString = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_DURATION
            )

            val durationMs = durationString?.toLongOrNull() ?: 0L

            if (durationMs <= 0L) {
                throw Exception("Could not determine video duration.")
            }

            val actualFrameCount =
                frameCount.coerceIn(1, 30)

            for (i in 0 until actualFrameCount) {

                val timeUs =
                    if (actualFrameCount == 1) {
                        0L
                    } else {
                        (durationMs * 1000L * i) /
                                (actualFrameCount - 1)
                    }

                val bitmap = retriever.getFrameAtTime(
                    timeUs,
                    MediaMetadataRetriever.OPTION_CLOSEST
                )

                if (bitmap != null) {
                    val jpegBytes = bitmapToJpeg(bitmap)
                    frames.add(jpegBytes)
                    bitmap.recycle()
                }
            }

            if (frames.isEmpty()) {
                throw Exception("No video frames could be extracted.")
            }

            return frames

        } finally {
            retriever.release()
        }
    }

    private fun bitmapToJpeg(bitmap: Bitmap): ByteArray {

        val outputStream = ByteArrayOutputStream()

        bitmap.compress(
            Bitmap.CompressFormat.JPEG,
            90,
            outputStream
        )

        return outputStream.toByteArray()
    }
}