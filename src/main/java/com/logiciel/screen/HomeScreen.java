/**
 * 
 */
package com.logiciel.screen;

import java.awt.Font;
import java.awt.GridBagLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingUtilities;

/**
 * 
 */
public class HomeScreen extends JFrame{
	private static final long serialVersionUID = 1L;

	public HomeScreen() {
		setTitle("Projet MAIA 2026");
		setSize(1536, 1024);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setLocationRelativeTo(null);
		ImageIcon backgroundIcon = new ImageIcon("accueil.png");
		JLabel backgroundLabel = new JLabel(backgroundIcon);
		backgroundLabel.setLayout(new GridBagLayout());
		JButton enterButton = new JButton("Entrer dans l'application");
		enterButton.setFont(new Font("Arial", Font.BOLD, 16));
		//
		enterButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				setSize(800, 600);
				setLocationRelativeTo(null);
				displayCNNScreen();
				}
		});
		backgroundLabel.setLayout(new BoxLayout(backgroundLabel, javax.swing.BoxLayout.Y_AXIS));
		enterButton.setAlignmentX(java.awt.Component.CENTER_ALIGNMENT);
		backgroundLabel.add(javax.swing.Box.createVerticalGlue());
		backgroundLabel.add(enterButton);
		backgroundLabel.add(Box.createVerticalStrut(163));
		
		add(backgroundLabel);
		//displayCNNScreen();
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SwingUtilities.invokeLater(() -> {
			HomeScreen app = new HomeScreen();
			app.setVisible(true);
		});
	}

	public void displaySRCScreen(String string) {
		// TODO Auto-generated method stub
		getContentPane().removeAll();
		ChoiceSRCScreen srcScreen = new ChoiceSRCScreen(this, string);
		add(srcScreen);
		revalidate();
		repaint();
	}

	public void displayCNNScreen() {
		//TODO Auto-generated method stub
		getContentPane().removeAll();
		ChoiceCNNScreen screen = new ChoiceCNNScreen(this);
		add(screen);
		revalidate();
		repaint();
	}
}
