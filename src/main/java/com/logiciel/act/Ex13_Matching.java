package com.logiciel.act;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.*;

import java.util.ArrayList;
import java.util.List;

public class Ex13_Matching {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // 1. Lecture de l'image de billard
        Mat billard = lectureImage("Images/Billard_Balls.jpg");
        ImShow("Billard", billard);

        // 2. Extraire les balles rouges (copie de la logique de Ex10)
        List<Mat> ballesRouges = extraireBallesRouges(billard);

        if (ballesRouges.size() < 2) {
            System.out.println("Moins de deux balles rouges détectées.");
            return;
        }

        Mat balleRouge1 = ballesRouges.get(0);
        Mat balleRouge2 = ballesRouges.get(1);

        ImShow("Balle rouge 1", balleRouge1);
        ImShow("Balle rouge 2", balleRouge2);

        // 3. Lecture de la balle 3 (template)
        Mat balle3 = lectureImage("Images/Ball_three.png");
        ImShow("Balle 3 (template)", balle3);

        // 4. Matching avec ratio test
        double score1 = matchORB(balle3, balleRouge1, "Matching balle 3 avec balle rouge 1");
        double score2 = matchORB(balle3, balleRouge2, "Matching balle 3 avec balle rouge 2");

        System.out.println("\n--- Scores finaux ---");
        System.out.println("Score balle rouge 1 : " + score1);
        System.out.println("Score balle rouge 2 : " + score2);

        if (score1 < score2) {
            System.out.println("=> La balle rouge 1 est la meilleure correspondance !");
            ImShow("Meilleure correspondance", balleRouge1);
        } else {
            System.out.println("=> La balle rouge 2 est la meilleure correspondance !");
            ImShow("Meilleure correspondance", balleRouge2);
        }
    }

    // -------- Extraction des balles rouges (copie de Ex10) --------
    public static List<Mat> extraireBallesRouges(Mat m) {

        List<Mat> ballesRouges = new ArrayList<>();

        Mat hsvImage = new Mat();
        Imgproc.cvtColor(m, hsvImage, Imgproc.COLOR_BGR2HSV);

        Mat threshold1 = new Mat();
        Mat threshold2 = new Mat();
        Mat threshold = new Mat();

        Core.inRange(hsvImage, new Scalar(0, 100, 100), new Scalar(10, 255, 255), threshold1);
        Core.inRange(hsvImage, new Scalar(160, 100, 100), new Scalar(179, 255, 255), threshold2);
        Core.bitwise_or(threshold1, threshold2, threshold);

        Imgproc.GaussianBlur(threshold, threshold, new Size(9, 9), 2, 2);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.Canny(threshold, threshold, 100, 200);
        Imgproc.findContours(threshold, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        MatOfPoint2f matOfPoint2f = new MatOfPoint2f();
        float[] radius = new float[1];
        Point center = new Point();

        for (MatOfPoint contour : contours) {

            double contourArea = Imgproc.contourArea(contour);
            if (contourArea < 50) continue;

            matOfPoint2f.fromList(contour.toList());
            Imgproc.minEnclosingCircle(matOfPoint2f, center, radius);

            double circleArea = Math.PI * radius[0] * radius[0];
            double ratio = contourArea / circleArea;

            if (ratio >= 0.8) {
                Rect rect = Imgproc.boundingRect(contour);
                Mat tmp = m.submat(rect.y, rect.y + rect.height, rect.x, rect.x + rect.width);
                Mat ball = new Mat(tmp.size(), tmp.type());
                tmp.copyTo(ball);
                ballesRouges.add(ball);
            }
        }

        return ballesRouges;
    }

    // -------- Matching ORB avec ratio test (Lowe) --------
    public static double matchORB(Mat template, Mat object, String windowName) {

        Mat scaledObject = new Mat();
        Imgproc.resize(object, scaledObject, template.size());

        Mat grayObject = new Mat();
        Mat grayTemplate = new Mat();
        Imgproc.cvtColor(scaledObject, grayObject, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(template, grayTemplate, Imgproc.COLOR_BGR2GRAY);

        Core.normalize(grayObject, grayObject, 0, 255, Core.NORM_MINMAX);
        Core.normalize(grayTemplate, grayTemplate, 0, 255, Core.NORM_MINMAX);

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);

        MatOfKeyPoint kpObj = new MatOfKeyPoint();
        MatOfKeyPoint kpTpl = new MatOfKeyPoint();
        detector.detect(grayObject, kpObj);
        detector.detect(grayTemplate, kpTpl);

        Mat descObj = new Mat();
        Mat descTpl = new Mat();
        extractor.compute(grayObject, kpObj, descObj);
        extractor.compute(grayTemplate, kpTpl, descTpl);

        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

        List<MatOfDMatch> knnMatches = new ArrayList<>();
        matcher.knnMatch(descObj, descTpl, knnMatches, 2);

        List<DMatch> goodMatches = new ArrayList<>();
        for (MatOfDMatch m : knnMatches) {
            DMatch[] d = m.toArray();
            if (d.length == 2 && d[0].distance < 0.75 * d[1].distance) {
                goodMatches.add(d[0]);
            }
        }

        double score = 0;
        for (DMatch m : goodMatches) {
            score += m.distance;
        }

        if (goodMatches.size() > 0) {
            score /= goodMatches.size();
        } else {
            score = Double.MAX_VALUE;
        }

        MatOfDMatch goodMat = new MatOfDMatch();
        goodMat.fromList(goodMatches);

        Mat matchedImage = new Mat();
        Features2d.drawMatches(scaledObject, kpObj, template, kpTpl, goodMat, matchedImage);
        ImShow(windowName, matchedImage);

        System.out.println(windowName + " — bons matches : " + goodMatches.size());
        System.out.println("Score = " + score);

        return score;
    }

    // -------- Lecture d'image --------
    public static Mat lectureImage(String fichier) {
        return Highgui.imread(fichier);
    }

    // -------- Affichage --------
    public static void ImShow(String title, Mat img) {
        MatOfByte mob = new MatOfByte();
        Highgui.imencode(".png", img, mob);
        byte[] ba = mob.toArray();
        try {
            java.io.InputStream in = new java.io.ByteArrayInputStream(ba);
            java.awt.image.BufferedImage buf = javax.imageio.ImageIO.read(in);
            javax.swing.JFrame frame = new javax.swing.JFrame(title);
            frame.getContentPane().add(new javax.swing.JLabel(new javax.swing.ImageIcon(buf)));
            frame.pack();
            frame.setVisible(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
