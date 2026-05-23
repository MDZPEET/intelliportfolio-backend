"use client";

import { useAuth } from "@clerk/nextjs";
import { usePathname } from "next/navigation";
import { useEffect } from "react";

export default function AuthRedirectWatcher() {
  const { isSignedIn, isLoaded } = useAuth();
  const pathname = usePathname();

  useEffect(() => {
    // ถ้าสถานะโหลดเสร็จ + ล็อกอินอยู่ + อยู่ในหน้าที่ไม่ควรอยู่ (หน้าแรกหรือหน้าล็อกอิน)
    if (isLoaded && isSignedIn && (pathname === "/login" || pathname === "/register")) {
      // ใช้ window.location.href แทน router.push เพื่อป้องกันอาการค้างจากการโหลด State ซ้อนกัน
      window.location.href = "/dashboard";
    }
  }, [isSignedIn, isLoaded, pathname]);

  return null;
}