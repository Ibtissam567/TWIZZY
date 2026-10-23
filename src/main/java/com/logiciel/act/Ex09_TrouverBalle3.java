package com.logiciel.act;

import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Ex09_TrouverBalle3 {

    public static Mat LectureImage(String fichier) {
        return Highgui.imread(fichier);
    }

    public static void ImShow(String title, Mat img) {
        MatOfByte mob = new MatOfByte();
        Highgui.imencode(".png", img, mob);
        byte[] ba = mob.toArray();
        try {
            JFrame frame = new JFrame(title);
            frame.add(new JLabel(new ImageIcon(javax.imageio.ImageIO.read(new ByteArrayInputStream(ba)))));
            frame.pack();
            frame.setVisible(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Détection du rouge
    public static Mat DetecterRouge(Mat hsv) {
        Mat t1 = new Mat();
        Mat t2 = new Mat();
        Mat threshold = new Mat();

        Core.inRange(hsv, new Scalar(0, 100, 100), new Scalar(10, 255, 255), t1);
        Core.inRange(hsv, new Scalar(160, 100, 100), new Scalar(179, 255, 255), t2);

        Core.bitwise_or(t1, t2, threshold);
        Imgproc.GaussianBlur(threshold, threshold, new Size(9, 9), 2, 2);

        return threshold;
    }

    public static void main(String[] args) {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Image contenant toutes les balles
        Mat m = LectureImage("Images/Billard_Balls.jpg");
        if (m.empty()) {
            System.out.println("ERREUR : image introuvable !");
            return;
        }

        // Template de la balle 3
        Mat template = LectureImage("Images/Ball_three.png");
        if (template.empty()) {
            System.out.println("ERREUR : template introuvable !");
            return;
        }

        // Conversion en HSV
        Mat hsv = new Mat();
        Imgproc.cvtColor(m, hsv, Imgproc.COLOR_BGR2HSV);

        // Seuillage rouge
        Mat threshold = DetecterRouge(hsv);

        // Détection des contours
        List<MatOfPoint> contours = new ArrayList<>();
        MatOfInt4 hierarchy = new MatOfInt4();
        Imgproc.findContours(threshold, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // ORB
        FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);

        // Préparation du template
        Mat grayTemplate = new Mat();
        Imgproc.cvtColor(template, grayTemplate, Imgproc.COLOR_BGR2GRAY);
        Core.normalize(grayTemplate, grayTemplate, 0, 255, Core.NORM_MINMAX);

        MatOfKeyPoint templateKP = new MatOfKeyPoint();
        detector.detect(grayTemplate, templateKP);

        Mat templateDesc = new Mat();
        extractor.compute(grayTemplate, templateKP, templateDesc);

        double meilleurScore = Double.MAX_VALUE;
        Rect meilleurRect = null;

        // Parcourir chaque balle rouge détectée
        for (MatOfPoint contour : contours) {

            Rect rect = Imgproc.boundingRect(contour);
            Mat ball = m.submat(rect);

            // Mise à l’échelle
            Mat scaledBall = new Mat();
            Imgproc.resize(ball, scaledBall, template.size());

            // Conversion en gris
            Mat grayBall = new Mat();
            Imgproc.cvtColor(scaledBall, grayBall, Imgproc.COLOR_BGR2GRAY);
            Core.normalize(grayBall, grayBall, 0, 255, Core.NORM_MINMAX);

            // Keypoints
            MatOfKeyPoint ballKP = new MatOfKeyPoint();
            detector.detect(grayBall, ballKP);

            Mat ballDesc = new Mat();
            extractor.compute(grayBall, ballKP, ballDesc);

            if (ballDesc.empty()) continue;

            // Matching
            MatOfDMatch matches = new MatOfDMatch();
            matcher.match(ballDesc, templateDesc, matches);

            // Score = distance moyenne
            double score = 0;
            for (DMatch m2 : matches.toArray()) score += m2.distance;
            score /= matches.toArray().length;

            // Garder la meilleure correspondance
            if (score < meilleurScore) {
                meilleurScore = score;
                meilleurRect = rect;
            }
        }

        // Dessiner la balle 3 trouvée
        if (meilleurRect != null) {
        	Core.rectangle(m,
                    new Point(meilleurRect.x, meilleurRect.y),
                    new Point(meilleurRect.x + meilleurRect.width, meilleurRect.y + meilleurRect.height),
                    new Scalar(0, 255, 0), 3);

            ImShow("Balle 3 détectée", m);
        }

        System.out.println("Traitement terminé.");
    }
}
