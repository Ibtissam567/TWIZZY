package com.logiciel.processing;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import java.util.Collections;

public class ProcLib{
	// ///////////////////////////////// //
	// /		Pour l'Acquisition		//
	// /////////////////////////////// //
	public static BufferedImage Mat2bufferedImage(Mat image) {
		MatOfByte bytemat = new MatOfByte();
		Highgui.imencode(".jpg", image, bytemat);
		byte[] bytes = bytemat.toArray();
		InputStream in = new ByteArrayInputStream(bytes);
		BufferedImage img = null;
		try {
			img = ImageIO.read(in);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return img;
	}
	//
	// ///////////////////////////////////////// //
	// /		Transformation de l'image		//
	// /////////////////////////////////////// //
	//
	//Charge l'image dans une variable
	public static Mat LectureImage(String fichier) {
        return Highgui.imread(fichier);
    }
	//
	//Matrice : image BGR -> Matrice : image HSV
	public static Mat transformeBGRversHSV(Mat matriceBGR){
		Mat matriceHSV=new Mat(matriceBGR.height(),matriceBGR.cols(),matriceBGR.type());
		Imgproc.cvtColor(matriceBGR,matriceHSV,Imgproc.COLOR_BGR2HSV);
		return matriceHSV;
	}
	//
	//Matrice : image HSV -> 3 canaux : H S V
	public static Vector<Mat> splitHSVChannels(Mat input) {
		Vector<Mat> channels = new Vector<Mat>(); 
		Core.split(input, channels);
		return channels;
	}
	//
	//String : nom de la fenetre, Matrice image -> void
	public static void afficheImage(String title, Mat img) throws IOException{
		MatOfByte matOfByte=new MatOfByte();
		Highgui.imencode(".png",img,matOfByte);
		byte[] byteArray=matOfByte.toArray();
		BufferedImage bufImage=null;
		InputStream in=new ByteArrayInputStream(byteArray);
		bufImage=ImageIO.read(in);
		JFrame frame=new JFrame();
		frame.setTitle(title);
		frame.getContentPane().add(new JLabel(new ImageIcon(bufImage)));
		frame.pack();
		frame.setVisible(true);
	}
	//
	// ///////////////////////// //
	// /		Tresholding		//
	// /////////////////////// //
	//
	//Matrice : image HSV,int:seuil bas, int:seul haut, int:seuil saturation
	// -> Mat : image noir et blanc 
	public static Mat seuillageRouge(Mat input, int seuilRougeOrange, int seuilRougeViolet,int seuilSaturation){
		//Only for red Hue 
		Vector<Mat> channels = splitHSVChannels(input);
		//Scalars
		Scalar scRougeViolet = new Scalar(seuilRougeViolet);
		Scalar scRougeOrange = new Scalar(seuilRougeOrange);
		Scalar scSaturation = new Scalar(seuilSaturation);
		//Matrix
		Mat MatrougeBas = new Mat();
		Mat MatrougeHaut = new Mat();
		Mat MatHue = new Mat();
		Mat MatSat = new Mat();
		//Tresholding Hue
		Core.compare(channels.get(0), scRougeViolet, MatrougeHaut, Core.CMP_GT);
		Core.compare(channels.get(0), scRougeOrange, MatrougeBas, Core.CMP_LT);
		Core.bitwise_or(MatrougeHaut, MatrougeBas, MatHue);
		//Tresholding Saturation
		Core.compare(channels.get(1), scSaturation, MatSat, Core.CMP_GT);;
		//Mask
		Mat Mask = new Mat();
		Core.bitwise_and(MatSat, MatHue, Mask);
		return Mask;
	}
	//
	//Matrice : image HSV,int:seuil bas, int:seul haut, int:seuil saturation, int:seuil Value
	// -> Mat : image noir et blanc 
	public static Mat seuillageRouge(Mat input, int seuilRougeOrange, int seuilRougeViolet,int seuilSaturation, int seuilValue){
		//Only for red Hue 
		Vector<Mat> channels = splitHSVChannels(input);
		//Scalars
		Scalar scRougeViolet = new Scalar(seuilRougeViolet);
		Scalar scRougeOrange = new Scalar(seuilRougeOrange);
		Scalar scSaturation = new Scalar(seuilSaturation);
		Scalar scValue = new Scalar(seuilValue);
		//Matrix
		Mat MatrougeBas = new Mat();
		Mat MatrougeHaut = new Mat();
		Mat MatHue = new Mat();
		Mat MatSat = new Mat();
		Mat MatVal = new Mat();
		//Tresholding Hue
		Core.compare(channels.get(0), scRougeViolet, MatrougeHaut, Core.CMP_GT);
		Core.compare(channels.get(0), scRougeOrange, MatrougeBas, Core.CMP_LT);
		Core.bitwise_or(MatrougeHaut, MatrougeBas, MatHue);
		//Tresholding Saturation AND Value
		Core.compare(channels.get(1), scSaturation, MatSat, Core.CMP_GT);
		Core.compare(channels.get(2), scValue, MatVal, Core.CMP_GT);
		Mat MatSV = new Mat();
		Core.bitwise_and(MatSat, MatVal, MatSV);
		//Mask
		Mat Mask = new Mat();
		Core.bitwise_and(MatSV, MatHue, Mask);
		return Mask;
	}
	//Matrice : image HSV,int:seuil bas, int:seul haut, int:seuil saturation
	// -> Mat : image noir et blanc 
	public static Mat seuillageCouleur(Mat input, int seuilBas, int seuilHaut,int seuilSaturation){
		//For every Hue EXCEPT red
		Vector<Mat> channels = splitHSVChannels(input);
		//Scalars
		Scalar scHaut = new Scalar(seuilHaut);
		Scalar scBas = new Scalar(seuilBas);
		Scalar scSaturation = new Scalar(seuilSaturation);
		//Matrix
		Mat MatBas = new Mat();
		Mat MatHaut = new Mat();
		Mat MatHue = new Mat();
		Mat MatSat = new Mat();
		//Tresholding Hue
		Core.compare(channels.get(0), scBas, MatHaut, Core.CMP_GT);
		Core.compare(channels.get(0), scHaut, MatBas, Core.CMP_LT);
		Core.bitwise_and(MatHaut, MatBas, MatHue);
		//Tresholding Saturation AND
		Core.compare(channels.get(1), scSaturation, MatSat, Core.CMP_GT);
		//Mask
		Mat Mask = new Mat();
		Core.bitwise_and(MatSat, MatHue, Mask);
		return Mask;
	}
	//Matrice : image HSV,int:seuil bas, int:seul haut, int:seuil saturation, int:seuil Value
	// -> Mat : image noir et blanc 
	public static Mat seuillageCouleur(Mat input, int seuilBas, int seuilHaut,int seuilSaturation,int seuilValue){
		//For every Hue EXCEPT red
		Vector<Mat> channels = splitHSVChannels(input);
		//Scalars
		Scalar scHaut = new Scalar(seuilHaut);
		Scalar scBas = new Scalar(seuilBas);
		Scalar scSaturation = new Scalar(seuilSaturation);
		Scalar scValue = new Scalar(seuilValue);
		//Matrix
		Mat MatBas = new Mat();
		Mat MatHaut = new Mat();
		Mat MatHue = new Mat();
		Mat MatSat = new Mat();
		Mat MatVal = new Mat();
		//Tresholding Hue
		Core.compare(channels.get(0), scBas, MatHaut, Core.CMP_GT);
		Core.compare(channels.get(0), scHaut, MatBas, Core.CMP_LT);
		Core.bitwise_and(MatHaut, MatBas, MatHue);
		//Tresholding Saturation AND Value
		Core.compare(channels.get(1), scSaturation, MatSat, Core.CMP_GT);
		Core.compare(channels.get(2), scValue, MatVal, Core.CMP_GT);
		Mat MatSV = new Mat();
		Core.bitwise_and(MatSat, MatVal, MatSV);
		//Mask
		Mat Mask = new Mat();
		Core.bitwise_and(MatSV, MatHue, Mask);
		return Mask;
	}
	//
	// ///////////////////////////// //
	// /		Extract Form		//
	// / ///////////////////////// //
	//Matrice : image saturee -> Liste : Contours connexes 
	public static List<MatOfPoint> ExtractContours(Mat input){
		//System.out.println("extract contour");
		int thresh = 100;
		Mat canny_output = new Mat();
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		MatOfInt4 hierarchy = new MatOfInt4();
		//image saturee -> image contours des saturations
		Imgproc.Canny( input, canny_output, thresh, thresh*2);
		//afficheImage("e : canny output", canny_output);
		//image contours -> liste de : MatOfPoint, tableau de coo (x,y)
		Imgproc.findContours( canny_output, contours, hierarchy,Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
		return contours;
	}
	//
	//Matrice : image brute, 1 contour de l'image
	//-> Matrice : image decoupee 
	public static Mat DetectCircle(Mat img,MatOfPoint contour) throws IOException{
		//System.out.println("detect form");
		MatOfPoint2f matOfPoint2f = new MatOfPoint2f();
		float[] radius = new float[1];
		Point center = new Point();
		Rect rect = Imgproc.boundingRect(contour);
		//calcul de l'air de la figure
		double contourArea = Imgproc.contourArea(contour);
		matOfPoint2f.fromList(contour.toList());
		//cherche le plus petit cercle entourant le contour
		Imgproc.minEnclosingCircle(matOfPoint2f, center, radius);
		//test du cercle
		//aire reel >? 80% de l'aire max theorique (cercle parfait) 
		if ((contourArea / (Math.PI*radius[0]*radius[0])) >=0.8) {
			//decoupage du rectangle avec le panneau detectee 
			Mat tmp = img.submat(rect.y,rect.y+rect.height,rect.x,rect.x+rect.width);
			Mat sign = Mat.zeros(tmp.size(),tmp.type());
			tmp.copyTo(sign);
			//afficheImage("circle detectform",sign);
			return sign;
		}
		return null;
	}
	
	public static Mat DetectTriangle(Mat img,MatOfPoint contour) {
		MatOfPoint2f matOfPoint2f = new MatOfPoint2f();
		MatOfPoint2f approxCurve = new MatOfPoint2f();
		float[] radius = new float[1];
		Point center = new Point();
		Rect rect = Imgproc.boundingRect(contour);
		double contourArea = Imgproc.contourArea(contour);


		matOfPoint2f.fromList(contour.toList());
		// Cherche le plus petit cercle entourant le contour
		Imgproc.minEnclosingCircle(matOfPoint2f, center, radius);
		if ((contourArea / (Math.PI*radius[0]*radius[0])) >=0.8) {
			return null;
		}else {
			Imgproc.approxPolyDP(matOfPoint2f, approxCurve, Imgproc.arcLength(matOfPoint2f, true) * 0.02, true);
			long total = approxCurve.total();
			if (total == 3 ) { // is triangle
				//System.out.println("Triangle");
				Point [] pt = approxCurve.toArray();
				Core.line(img, pt[0], pt[1], new Scalar(255,0,0),2);
				Core.line(img, pt[1], pt[2], new Scalar(255,0,0),2);
				Core.line(img, pt[2], pt[0], new Scalar(255,0,0),2);
				Core.rectangle(img, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), new Scalar (0, 255, 0), 2);
				Mat tmp = img.submat(rect.y,rect.y+rect.height,rect.x,rect.x+rect.width);
				Mat sign = Mat.zeros(tmp.size(),tmp.type());
				tmp.copyTo(sign);
				return sign;
			}
			if (total >= 4 && total <= 6) {
				List<Double> cos = new ArrayList<>();
				Point[] points = approxCurve.toArray();
				for (int j = 2; j < total + 1; j++) {
					cos.add(angle(points[(int) (j % total)], points[j - 2], points[j - 1]));
				}
				Collections.sort(cos);
				Double minCos = cos.get(0);
				Double maxCos = cos.get(cos.size() - 1);
				boolean isRect = total == 4 && minCos >= -0.1 && maxCos <= 0.3;
				boolean isPolygon = (total == 5 && minCos >= -0.34 && maxCos <= -0.27) || (total == 6 && minCos >= -0.55 && maxCos <= -0.45);
				if (isRect) {
					double ratio = Math.abs(1 - (double) rect.width / rect.height);
					//drawText(rect.tl(), ratio <= 0.02 ? "SQU" : "RECT");
					//System.out.println("Rectangle");
					Core.rectangle(img, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), new Scalar (0, 255, 0), 2);
					Mat tmp = img.submat(rect.y,rect.y+rect.height,rect.x,rect.x+rect.width);
					Mat sign = Mat.zeros(tmp.size(),tmp.type());
					tmp.copyTo(sign);
					return null;
				}
				if (isPolygon) {
					//System.out.println("Polygon");
					//drawText(rect.tl(), "Polygon");
				}
			}			
		}
		return null;

	}
	//
	public static Mat DetectPolygon(Mat img,MatOfPoint contour){
		MatOfPoint2f matOfPoint2f = new MatOfPoint2f();
		MatOfPoint2f approxCurve = new MatOfPoint2f();
		float[] radius = new float[1];
		Point center = new Point();
		Rect rect = Imgproc.boundingRect(contour);
		double contourArea = Imgproc.contourArea(contour);


		matOfPoint2f.fromList(contour.toList());
		// Cherche le plus petit cercle entourant le contour
		Imgproc.minEnclosingCircle(matOfPoint2f, center, radius);
		if ((contourArea / (Math.PI*radius[0]*radius[0])) >=0.8) {
			return null;
		}else {
			Imgproc.approxPolyDP(matOfPoint2f, approxCurve, Imgproc.arcLength(matOfPoint2f, true) * 0.02, true);
			long total = approxCurve.total();
			if (total == 3 ) { // is triangle
				//System.out.println("Triangle");
				Point [] pt = approxCurve.toArray();
				Core.line(img, pt[0], pt[1], new Scalar(255,0,0),2);
				Core.line(img, pt[1], pt[2], new Scalar(255,0,0),2);
				Core.line(img, pt[2], pt[0], new Scalar(255,0,0),2);
				Core.rectangle(img, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), new Scalar (0, 255, 0), 2);
				Mat tmp = img.submat(rect.y,rect.y+rect.height,rect.x,rect.x+rect.width);
				Mat sign = Mat.zeros(tmp.size(),tmp.type());
				tmp.copyTo(sign);
				return null;
			}
			if (total >= 4 && total <= 6) {
				List<Double> cos = new ArrayList<>();
				Point[] points = approxCurve.toArray();
				for (int j = 2; j < total + 1; j++) {
					cos.add(angle(points[(int) (j % total)], points[j - 2], points[j - 1]));
				}
				Collections.sort(cos);
				Double minCos = cos.get(0);
				Double maxCos = cos.get(cos.size() - 1);
				boolean isRect = total == 4 && minCos >= -0.1 && maxCos <= 0.3;
				boolean isPolygon = (total == 5 && minCos >= -0.34 && maxCos <= -0.27) || (total == 6 && minCos >= -0.55 && maxCos <= -0.45);
				if (isRect) {
					double ratio = Math.abs(1 - (double) rect.width / rect.height);
					//drawText(rect.tl(), ratio <= 0.02 ? "SQU" : "RECT");
					//System.out.println("Rectangle");
					Core.rectangle(img, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), new Scalar (0, 255, 0), 2);
					Mat tmp = img.submat(rect.y,rect.y+rect.height,rect.x,rect.x+rect.width);
					Mat sign = Mat.zeros(tmp.size(),tmp.type());
					tmp.copyTo(sign);
					return sign;
				}
				if (isPolygon) {
					//System.out.println("Polygon");
					//drawText(rect.tl(), "Polygon");
				}
			}			
		}
		return null;

	}
	//
	//Matrice : image brute, 1 contour de l'image
	//-> Matrice : image decoupee 
	public static double angle(Point a, Point b, Point c){
		Point ab = new Point( b.x - a.x, b.y - a.y );
		Point cb = new Point( b.x - c.x, b.y - c.y );
		double dot = (ab.x * cb.x + ab.y * cb.y); // dot product
		double cross = (ab.x * cb.y - ab.y * cb.x); // cross product
		double alpha = Math.atan2(cross, dot);
		return Math.floor(alpha * 180. / Math.PI + 0.5);
	}
	// ///////////////////////////////////////// //
	// /		Reconaissance du panneau		//
	// / ///////////////////////////////////// //
	//a renforcer/remplacer par un CNN 
	//Matrice : ,String nom de l'image de reference brute
	//->
	public static double Similitude(Mat object,String signfile){
		// Conversion du signe de reference en niveaux de gris et normalisation
		Mat panneauref = Highgui.imread(signfile);
		Mat graySign = new Mat(panneauref.rows(), panneauref.cols(), panneauref.type());
		Imgproc.cvtColor(panneauref, graySign, Imgproc.COLOR_BGRA2GRAY);
		//MaBibliothequeTraitementImage.afficheImage("GraySign ref",graySign);
		Core.normalize(graySign, graySign, 0, 255, Core.NORM_MINMAX);
		//MaBibliothequeTraitementImage.afficheImage("M GraySign ref",graySign);
		//Mat signeNoirEtBlanc=new Mat();
		//Conversion du panneau extrait de l'image en gris et normalisation et redimensionnement de la taille du panneau de reference
		Mat grayObject = new Mat(panneauref.rows(), panneauref.cols(), panneauref.type());
		Imgproc.resize(object, object, graySign.size());
		Imgproc.cvtColor(object, grayObject, Imgproc.COLOR_BGRA2GRAY);
		Core.normalize(grayObject, grayObject, 0, 255, Core.NORM_MINMAX);
		//Imgproc.resize(grayObject, grayObject, graySign.size());	
		Mat diff = new Mat();
		Core.absdiff(grayObject, graySign, diff);
		Scalar somme = Core.sumElems(diff);
		double totalDiff = somme.val[0];
		double moyDiff = totalDiff/(graySign.rows()*graySign.cols());
		return (255-moyDiff)/255;
	}
	
	
	
	
	
	public static boolean isInsideFrame(Mat img, Rect rect) {
		return (rect.x >= 0 && rect.y >= 0 &&
				rect.x + rect.width <= img.cols() &&
				rect.y + rect.height <= img.rows());
	}
	public static boolean isInsideFrame(Mat img, Rect rect,int delta) {
		return ((rect.x-delta) >= 0 && (rect.y-delta) >= 0 &&
				(rect.x + rect.width+delta) <= img.cols() &&
				(rect.y + rect.height+delta) <= img.rows());
	}
	//
	// ///////////////////////// //
	// /		Extraction		//
	// /////////////////////// //
	//
	public static List<List<Mat>> detectAllForm(Mat img, Mat imgSat) throws IOException {
	    //Initiating Lists
	    List<List<Mat>> globalResult = new ArrayList<>();
	    List<Mat> CircleList = new ArrayList<>();
	    List<Mat> TriangleList = new ArrayList<>();
	    List<Mat> PolygonList = new ArrayList<>();
	    //Cloning the first mask imgSat for Circle detection
	    Mat maskCopy = imgSat.clone();
	    // ///////////////////////////////////////////// //
	    // /		Step 1 : Circles with Hough			//
	    // /////////////////////////////////////////// //
	    Mat circles = new Mat();
	    Mat blurred = new Mat();
	    Mat masqueLisse = new Mat();
	    Mat elementNettoyage = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new org.opencv.core.Size(3, 3));
	    Imgproc.morphologyEx(maskCopy, masqueLisse, Imgproc.MORPH_OPEN, elementNettoyage);
	    Imgproc.morphologyEx(masqueLisse, masqueLisse, Imgproc.MORPH_CLOSE, elementNettoyage);
	    
	    //To avoid non-connexe Circles 
	    int blur = 3;
	    // DEBUG 1
	    //System.out.println("--- DIAGNOSTIC MATRICE VIDEO ---");
	    //System.out.println("Nombre de canaux (Channels) : " + maskCopy.channels());
	    //System.out.println("Type OpenCV (ex: 0 pour CV_8UC1) : " + maskCopy.type());
	    //System.out.println("Dimensions : " + maskCopy.cols() + "x" + maskCopy.rows());
	    Imgproc.GaussianBlur(masqueLisse, blurred, new org.opencv.core.Size(blur, blur),0,0);
	    //Hough detection 
	    //min dist between 2 circles
	    int minDiffR = blurred.rows()/8;
	    //minimal gradien edge
	    int diffEdge = 100; 
	    //treshold : cercle "minimal" 2pi*19 + 2pi*15 ~ 213px 
	    // environ 2/3 de vote pour une detection, ~142
	    // ajustement pour un seuil plus robuste
	    int treshold = 56;
	    //Ratio Hough space deformation, DO NOT ask how 1.73 0_° I mean it
	    double ratioH = 1.73;
	    //min radius, max radius 13 150
	    int minR = 15; int maxR = 250;
	    //
	    //afficheImage("blur",blurred);
	    Imgproc.HoughCircles(blurred, circles, Imgproc.CV_HOUGH_GRADIENT, ratioH , minDiffR, diffEdge, treshold, minR, maxR);
	    //System.out.println("DEBUG POST-LISSAGE : Cercles trouvés = " + circles.cols());
	    //Check if theses circle Actually in the image 
	    for (int x = 0; x < circles.cols(); x++) {
	        double[] c = circles.get(0, x);
	        Point center = new Point(Math.round(c[0]), Math.round(c[1]));
	        int radius = (int) Math.round(c[2]);
	        Rect rect = new Rect((int)(center.x - radius), (int)(center.y - radius), radius * 2, radius * 2);
	        //If it does not cross the image's borders 
	        //System.out.println("valid");
	        int delta = 5;
	        if (isInsideFrame(img,rect,delta)){
	        	//System.out.println("yes valid");
	            Mat tmp = img.submat(rect.y-delta, rect.y + rect.height+delta, rect.x-delta, rect.x + rect.width+delta);
	            Mat sign = Mat.zeros(tmp.size(), tmp.type());
	            tmp.copyTo(sign);
	            CircleList.add(sign);
	        }
	        else if (isInsideFrame(img,rect)) {
	        	Mat tmp = img.submat(rect.y, rect.y + rect.height, rect.x, rect.x + rect.width);
	            Mat sign = Mat.zeros(tmp.size(), tmp.type());
	            tmp.copyTo(sign);
	            CircleList.add(sign);
	        }
	        //Circle are convex regular Polygon with infinite amount of corners
	        //Removing theses Circle from the mask 
	        //To avoid detecting them as Polygons
	        Core.circle(maskCopy, center, radius + 10, new Scalar(0), -1);
	    }
	    //
	    // ///////////////////////////////// //
	    // /		Step 3 : Filter			//
	    // /////////////////////////////// //
	    //To connect non-convex part of an area
	    Mat masqueRes = new Mat();
	    //
	    int sizeElement = 5;
	    Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(sizeElement, sizeElement));
	    // MORPH_CLOSE réalise une Dilatation (grossir) puis une Érosion (revenir à la taille initiale)
	    Imgproc.morphologyEx(maskCopy, masqueRes, Imgproc.MORPH_CLOSE, element);
	    //afficheImage("res",masqueRes);
	    // ///////////////////////////////////////////////////// //
	    // /		Step 4 : Triangles and others Polygons		//
	    // /////////////////////////////////////////////////// //
	    //Edges detection 
	    List<MatOfPoint> contours = new ArrayList<>();
	    MatOfInt4 hierarchy = new MatOfInt4();
	    Imgproc.findContours(masqueRes, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
	    //
	    for (MatOfPoint contour : contours) {
	        //Noise/Range filtering , Triangle B = 20, H = 15 ~150px
	        if (Imgproc.contourArea(contour) < 250) continue;
	        MatOfPoint2f matOfPoint2f = new MatOfPoint2f(contour.toArray());
	        MatOfPoint2f approxCurve = new MatOfPoint2f();
	        Rect rect = Imgproc.boundingRect(contour);

	        //Corner Tolerance
	        double tolerance = 0.1;
	        //double epsilon = tolerance*Imgproc.arcLength(contour2f, true);
	        Imgproc.approxPolyDP(matOfPoint2f, approxCurve, Imgproc.arcLength(matOfPoint2f, true) *tolerance, true);
	        long total = approxCurve.total();

	        if (isInsideFrame(img,rect)) {
	            Mat tmp = img.submat(rect.y, rect.y + rect.height, rect.x, rect.x + rect.width);
	            Mat sign = Mat.zeros(tmp.size(), tmp.type());
	            tmp.copyTo(sign);

	            //Triangle
	            if (total == 3) { 
	            	TriangleList.add(sign);
	            }
	            
	            //
	            else if (total >= 4) {
	                List<Double> cos = new ArrayList<>();
	                Point[] points = approxCurve.toArray();
	                for (int j = 2; j < total + 1; j++) {
	                    cos.add(angle(points[(int) (j % total)], points[j - 2], points[j - 1]));
	                }
	                java.util.Collections.sort(cos);
	                Double minCos = cos.get(0);
	                Double maxCos = cos.get(cos.size() - 1);
	                
	                boolean isRect = total == 4 && minCos >= -0.1 && maxCos <= 0.3;
	                
	                if (isRect) {
	                	PolygonList.add(sign);
	                }
	            }
	        }
	    }
	    //
	    globalResult.add(CircleList);		//i=0
	    globalResult.add(TriangleList);		//i=1
	    globalResult.add(PolygonList);		//i=2
	    return globalResult;
	}
public static List<Mat> subExtractForm(Mat img, Mat imgSat,int ind) throws IOException{
	List<List<Mat>> list = detectAllForm(img,imgSat);
	return list.get(ind);
}

public static List<List<Mat>> extractAll(Mat img) throws IOException{
	Mat frameHSV = transformeBGRversHSV(img);
	Mat frameSat = seuillageRouge(frameHSV, 6, 170, 110,35);
	List<List<Mat>> res = detectAllForm(img,frameSat);
	return res;
}

public static List<Mat> extractCircles(Mat img) throws IOException{
	Mat frameHSV = transformeBGRversHSV(img);
	Mat frameSat = seuillageRouge(frameHSV, 6, 170, 110,35);
	List<Mat> res = subExtractForm(img,frameSat,0);
	return res;
}
}